pub mod brushes;
mod terrain;
mod utils;

use std::ops::{DerefMut, RangeFrom};

use bevy::{
    color::palettes::css::WHITE,
    pbr::{ExtendedMaterial, MaterialExtension},
    picking::{backend::ray::RayMap, mesh_picking::ray_cast::RayMeshHit},
    prelude::*,
    render::render_resource::{self, AsBindGroup},
};
use bevy_flycam::{MovementSettings, PlayerPlugin};
use bevy_inspector_egui::quick::WorldInspectorPlugin;
use brushes::{apply_brush, BrushKind};
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use terrain::{Chunk, Terrain};
use utils::parse_heightmap;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(PlayerPlugin)
        .add_plugins(WorldInspectorPlugin::new())
        .add_systems(Startup, setup)
        .add_event::<RecalculateNormals>()
        .add_systems(
            Update,
            (
                change_brush_kind,
                create_array_texture,
                raycast.pipe(raycast_handle).pipe(pipe_noop_option),
            ),
        )
        .add_systems(Update, recalculate_normals_for_chunk.pipe(pipe_noop_option))
        .insert_resource(BrushConfig::default())
        .register_type::<BrushConfig>()
        .register_type::<ChunkMetadata>()
        .insert_resource(MovementSettings {
            sensitivity: 0.00015,
            speed: 60.0,
        })
        .add_plugins(MaterialPlugin::<
            ExtendedMaterial<StandardMaterial, TerrainExtension>,
        >::default())
        .run();
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct TerrainExtension {
    #[texture(100, dimension = "2d_array")]
    #[sampler(101)]
    array_texture: Handle<Image>,

    #[texture(102, dimension = "2d", sample_type = "u_int")]
    #[sampler(103, sampler_type = "non_filtering")]
    material_index_map: Handle<Image>,

    #[uniform(104)]
    layers: u32,
}

impl MaterialExtension for TerrainExtension {
    fn fragment_shader() -> render_resource::ShaderRef {
        "shaders/terrain_shader.wgsl".into()
    }

    fn deferred_fragment_shader() -> render_resource::ShaderRef {
        "shaders/terrain_shader.wgsl".into()
    }
}

#[derive(Resource)]
struct LoadingTerrainTexture {
    is_loaded: bool,
    array_texture: Handle<Image>,
    material_index_map: Handle<Image>,
}

#[derive(Debug, Component, PartialEq, Eq, Reflect)]
#[reflect(Component)]
pub struct ChunkMetadata {
    x: usize,
    y: usize,
}

impl ChunkMetadata {
    pub fn new(x: usize, y: usize) -> Self {
        Self { x, y }
    }

    pub fn vec2(&self) -> Vec2 {
        Vec2::new(self.x as f32, self.y as f32)
    }
}

fn create_array_texture(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut loading_texture: ResMut<LoadingTerrainTexture>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut terrain_materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, TerrainExtension>>>,
) {
    let is_textures_are_loaded = asset_server
        .load_state(&loading_texture.material_index_map)
        .is_loaded()
        && asset_server
            .load_state(&loading_texture.array_texture)
            .is_loaded();

    if loading_texture.is_loaded || !is_textures_are_loaded {
        return;
    }
    loading_texture.is_loaded = true;
    let mut layers = 0;

    if let Some(image) = images.get_mut(&loading_texture.array_texture) {
        layers = image.height() / image.width();
        image.reinterpret_stacked_2d_as_array(layers);
    }

    if let Some(image) = images.get_mut(&loading_texture.material_index_map) {
        image.texture_descriptor.format = render_resource::TextureFormat::Rgba8Uint;
        image.sampler = bevy::image::ImageSampler::nearest();
    }

    let extension_handle = terrain_materials.add(ExtendedMaterial {
        base: StandardMaterial {
            base_color: Color::NONE,
            ..Default::default()
        },
        extension: TerrainExtension {
            array_texture: loading_texture.array_texture.clone(),
            material_index_map: loading_texture.material_index_map.clone(),
            layers,
        },
    });

    let heightmap = parse_heightmap("assets/heightmap2.png");
    let num_chunks = heightmap.len() / Chunk::SIZE;

    let terrain = Terrain::generate_chunks(&heightmap, num_chunks);

    for chunk in &terrain.chunks {
        commands.spawn((
            Name::new(format!("Chunk {}x{}", chunk.chunk_x, chunk.chunk_y)),
            Mesh3d(meshes.add(chunk.clone().into_mesh())),
            ChunkMetadata::new(chunk.chunk_x, chunk.chunk_y),
            MeshMaterial3d(extension_handle.clone()),
            Transform::from_xyz(
                (chunk.chunk_x * Chunk::SIZE) as f32,
                0.0,
                (chunk.chunk_y * Chunk::SIZE) as f32,
            ),
        ));
    }
}

#[derive(Debug, Resource, Reflect)]
#[reflect(Resource)]
pub struct BrushConfig {
    pub range: f32,
    pub strength: f32,
    pub kind: BrushKind,
    /// 0.1 to 0.5: Gentle falloff (spread effect widely).
    /// 1.0: Linear falloff (balanced).
    /// 2.0 to 5.0: Steep falloff (concentrated near the center).
    #[reflect(@RangeFrom::<f32> { start: 0.0 })]
    pub falloff_exponent: f32,
    pub flatten_level: Option<f32>,
}

impl BrushConfig {
    #[inline]
    pub fn range_squared(&self) -> f32 {
        self.range * self.range
    }

    #[inline]
    pub fn falloff_exponent(&self) -> f32 {
        self.falloff_exponent.max(0.1)
    }
}

impl Default for BrushConfig {
    fn default() -> Self {
        Self {
            range: 30.0,
            strength: 1.0,
            kind: BrushKind::Sculp,
            falloff_exponent: 1.0,
            flatten_level: None,
        }
    }
}

fn raycast(
    mut ray_cast: MeshRayCast,
    ray_map: Res<RayMap>,
    keys: Res<ButtonInput<MouseButton>>,
    mut config: ResMut<BrushConfig>,
    mut gizmos: Gizmos,
) -> Option<(Entity, RayMeshHit)> {
    for (_, ray) in ray_map.iter() {
        let Some((entity, hit)) = ray_cast.cast_ray(*ray, &RayCastSettings::default()).first()
        else {
            continue;
        };

        gizmos
            .circle(
                Isometry3d::new(hit.point, Quat::from_rotation_arc(Vec3::Z, Vec3::Y)),
                config.range,
                bevy::color::palettes::css::NAVY,
            )
            .resolution(64);

        if !keys.pressed(MouseButton::Left) {
            config.flatten_level = None;
            return None;
        }

        return Some((*entity, hit.clone()));
    }

    None
}

fn raycast_handle(
    In(maybe_hit): In<Option<(Entity, RayMeshHit)>>,
    mesh_3d: Query<(&GlobalTransform, &Mesh3d)>,
    metadata: Query<(Entity, &ChunkMetadata)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut config: ResMut<BrushConfig>,
    mut recalculate_normals: EventWriter<RecalculateNormals>,
) -> Option<()> {
    let (entity, hit) = maybe_hit?;
    let (chunk_gt, _) = mesh_3d.get(entity).ok()?;
    let center_position = chunk_gt.affine().inverse().transform_point3(hit.point);
    let center_vec2 = hit.point.xz();

    if matches!(config.kind, BrushKind::Flatten) && config.flatten_level.is_none() {
        config.flatten_level = Some(center_position.y);
    }

    let grid_size = (metadata.iter().count() as f32).sqrt() as usize;
    let chunk_size_f32 = Chunk::SIZE as f32;
    let half_square = chunk_size_f32 / 2.0;
    let half_diagonal = (half_square * 2.0_f32.sqrt()).ceil();
    let radius = config.range;

    let entities = (0..grid_size)
        .flat_map(|ref i| {
            (0..grid_size)
                .filter_map(|j| {
                    let square_center = (Vec2::new(*i as f32, j as f32) + 0.5) * chunk_size_f32;
                    let distance = (center_vec2 - square_center).length_squared().sqrt();

                    if distance <= radius + half_diagonal {
                        metadata.iter().find_map(|(e, metadata)| {
                            ChunkMetadata::new(*i, j).eq(metadata).then_some(e)
                        })
                    } else {
                        None
                    }
                })
                .collect_vec()
        })
        .collect_vec();

    let meshes_ptr: *mut Assets<Mesh> = meshes.deref_mut();

    entities
        .iter()
        .filter_map(|entity| {
            let (chunk_gt, chunk) = mesh_3d.get(*entity).ok()?;
            let center = chunk_gt.affine().inverse().transform_point3(hit.point).xz();

            // SAFETY: Each chunk mesh is unique to each chunk entity
            let chunk_mesh = unsafe {
                let meshes = &mut *meshes_ptr;
                meshes.get_mut(&chunk.0)?
            };
            Some((center, chunk_mesh))
        })
        .collect_vec()
        .into_par_iter()
        .map(|(center, chunk_mesh)| apply_brush(center, chunk_mesh, &config))
        .collect::<Option<Vec<_>>>()?;

    recalculate_normals.send_batch(entities.into_iter().map(RecalculateNormals));

    Some(())
}

#[derive(Debug, Event, PartialEq, Eq, Hash, Deref)]
pub struct RecalculateNormals(Entity);

// TODO: this is kinda fucked up, but fine for now
fn recalculate_normals_for_chunk(
    mut recalculate_normals: EventReader<RecalculateNormals>,
    mesh_3d: Query<&Mesh3d>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut local_timer: Local<Timer>,
    time: Res<Time>,
) -> Option<()> {
    if local_timer.duration().as_nanos() == 0 {
        local_timer.set_duration(std::time::Duration::from_secs_f32(0.05));
        local_timer.set_mode(TimerMode::Repeating);
        return Some(());
    }
    if local_timer.finished() {
        local_timer.reset();
    } else {
        local_timer.tick(time.delta());
        return Some(());
    }

    let chunks_entities = recalculate_normals
        .read()
        .unique()
        .map(std::ops::Deref::deref)
        .collect_vec();

    for entity in chunks_entities {
        let chunk = mesh_3d.get(*entity).ok()?;
        let chunk_mesh = meshes.get_mut(&chunk.0)?;
        chunk_mesh.compute_normals();
    }

    Some(())
}

fn change_brush_kind(mut config: ResMut<BrushConfig>, keys: Res<ButtonInput<KeyCode>>) {
    if keys.just_pressed(KeyCode::Digit1) {
        config.kind = BrushKind::Sculp;
    }

    if keys.just_pressed(KeyCode::Digit2) {
        config.kind = BrushKind::Flatten;
    }

    if keys.just_pressed(KeyCode::Digit3) {
        config.kind = BrushKind::Smooth;
    }

    if keys.pressed(KeyCode::PageUp) {
        config.range += 0.25;
    }

    if keys.pressed(KeyCode::PageDown) {
        config.range -= 0.25;
    }
}

fn pipe_noop_option<T>(_: In<Option<T>>) {}

fn setup(
    asset_server: Res<AssetServer>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut config_store: ResMut<GizmoConfigStore>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
) {
    for (_, config, _) in config_store.iter_mut() {
        config.depth_bias = -1.;
    }

    commands.insert_resource(LoadingTerrainTexture {
        is_loaded: false,
        array_texture: asset_server.load("textures/array_texture.png"),
        material_index_map: asset_server.load("textures/custom_map.png"),
    });

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(standard_materials.add(Color::srgb_u8(124, 144, 255))),
        Transform::from_xyz(0.0, 0.5, 0.0),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 10_000.0,
            color: WHITE.into(),
            ..default()
        },
        Transform::from_xyz(5.0, 5.0, 5.0).with_rotation(Quat::from_euler(
            EulerRot::XYZ,
            -0.1,
            0.0,
            0.0,
        )),
    ));
}
