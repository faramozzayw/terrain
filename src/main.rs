mod terrain;
mod utils;

use std::{collections::HashMap, ops::RangeFrom};

use bevy::{
    color::palettes::css::WHITE,
    pbr::{ExtendedMaterial, MaterialExtension},
    picking::{backend::ray::RayMap, mesh_picking::ray_cast::RayMeshHit},
    prelude::*,
    render::render_resource::{self, AsBindGroup},
};
use bevy_flycam::{MovementSettings, PlayerPlugin};
use bevy_inspector_egui::quick::WorldInspectorPlugin;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use terrain::{Chunk, Terrain};
use utils::parse_heightmap;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(PlayerPlugin)
        .add_plugins(WorldInspectorPlugin::new())
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                change_brush_kind,
                create_array_texture,
                raycast.pipe(raycast_handle).pipe(pipe_noop_option),
            ),
        )
        .insert_resource(Config::default())
        .register_type::<Config>()
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
    let num_chunks = heightmap.len() / Chunk::CHUNK_SIZE;

    let terrain = Terrain::generate_chunks(&heightmap, num_chunks);

    for chunk in &terrain.chunks {
        let x = (chunk.chunk_x * Chunk::CHUNK_SIZE) as f32;
        let z = (chunk.chunk_y * Chunk::CHUNK_SIZE) as f32;

        commands.spawn((
            Name::new(format!("Chunk [{x},{z}]")),
            Mesh3d(meshes.add(chunk.clone().into_mesh())),
            MeshMaterial3d(extension_handle.clone()),
            Transform::from_xyz(x, 0.0, z),
        ));
    }
}

fn raycast(
    mut ray_cast: MeshRayCast,
    mut _gizmos: Gizmos,
    ray_map: Res<RayMap>,
    keys: Res<ButtonInput<MouseButton>>,
) -> Option<(Entity, RayMeshHit)> {
    if !keys.pressed(MouseButton::Left) {
        return None;
    }

    for (_, ray) in ray_map.iter() {
        let Some((entity, hit)) = ray_cast.cast_ray(*ray, &RayCastSettings::default()).first()
        else {
            continue;
        };

        return Some((*entity, hit.clone()));
    }

    None
}

#[derive(Debug, Resource, Reflect)]
#[reflect(Resource)]
pub struct Config {
    pub range: f32,
    pub strength: f32,
    pub kind: BrushKind,
    /// 0.1 to 0.5: Gentle falloff (spread effect widely).
    /// 1.0: Linear falloff (balanced).
    /// 2.0 to 5.0: Steep falloff (concentrated near the center).
    #[reflect(@RangeFrom::<f32> { start: 0.0 })]
    pub falloff_exponent: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            range: 20.0,
            strength: 1.0,
            kind: BrushKind::Sculp,
            falloff_exponent: 1.0,
        }
    }
}

#[derive(Debug, Reflect, Clone, Copy)]
pub enum BrushKind {
    Sculp,
    Flatten,
    Smooth,
}

fn raycast_handle(
    In(maybe_hit): In<Option<(Entity, RayMeshHit)>>,
    mesh_3d: Query<(&GlobalTransform, &Mesh3d)>,
    mut meshes: ResMut<Assets<Mesh>>,
    config: Res<Config>,
) -> Option<()> {
    let (entity, hit) = maybe_hit?;
    let (chunk_gt, chunk) = mesh_3d.get(entity).ok()?;
    let chunk_mesh = meshes.get_mut(&chunk.0)?;
    let positions = match chunk_mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION) {
        Some(bevy::render::mesh::VertexAttributeValues::Float32x3(positions)) => positions,
        _ => return None,
    };

    let center_position = chunk_gt.affine().inverse().transform_point3(hit.point);
    let center = center_position.xz();
    let Config {
        range,
        strength,
        kind,
        falloff_exponent,
        ..
    } = *config;
    let falloff_exponent = falloff_exponent.max(0.1);

    let radius_squared = range * range;
    let snapshot = positions.clone();
    let mut grid: HashMap<(i32, i32), Vec<&[f32; 3]>> = HashMap::with_capacity(snapshot.len());

    snapshot.iter().for_each(|pos| {
        let cell = (
            (pos[0] / range).floor() as i32,
            (pos[2] / range).floor() as i32,
        );
        grid.entry(cell).or_default().push(pos);
    });

    positions
        .par_iter_mut()
        .filter_map(|position| {
            let this = Vec2::new(position[0], position[2]);
            let distance_squared = center.distance_squared(this);

            if distance_squared <= radius_squared {
                let distance = distance_squared.sqrt();
                let normalized_distance = (distance / range).min(1.0); // Normalize to [0, 1]
                let falloff = (1.0 - normalized_distance).powf(falloff_exponent);

                Some((position, falloff))
            } else {
                None
            }
        })
        .for_each(|(position, falloff)| match kind {
            BrushKind::Sculp => {
                position[1] += strength * falloff;
            }
            BrushKind::Flatten => {
                position[1] = position[1] * (1.0 - falloff) + center_position.y * falloff;
            }
            BrushKind::Smooth => {
                let this = Vec2::new(position[0], position[2]);

                let cell = (
                    (this.x / range).floor() as i32,
                    (this.y / range).floor() as i32,
                );

                // This code generates a 3x3 grid of neighboring cells around a target cell `(cell.0, cell.1)`.
                // The range `(-1..=1)` produces offsets for both the x and z axes, covering the target cell and its 8 neighbors.
                // For each `(dx, dz)` pair, we compute the new coordinates `(cell.0 + dx, cell.1 + dz)` and return them.
                // The result is a list of all neighboring cells (including the target cell itself) within a 3x3 grid.
                let neighbors = (-1..=1)
                    .flat_map(|dx| (-1..=1).map(move |dz| (cell.0 + dx, cell.1 + dz)))
                    .flat_map(|neighbor_cell| grid.get(&neighbor_cell).into_iter().flatten());

                let mut sum_heights = 0.0;
                let mut count = 0;

                neighbors.for_each(|neighbor| {
                    let neighbor_pos = Vec2::new(neighbor[0], neighbor[2]);
                    if this.distance_squared(neighbor_pos) <= radius_squared {
                        sum_heights += neighbor[1];
                        count += 1;
                    }
                });

                if count > 0 {
                    let smoothed_height = sum_heights / count as f32;
                    position[1] = position[1] * (1.0 - strength * falloff)
                        + smoothed_height * (strength * falloff);
                }
            }
        });

    chunk_mesh.compute_normals();

    Some(())
}

fn change_brush_kind(mut config: ResMut<Config>, keys: Res<ButtonInput<KeyCode>>) {
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
        config.range += 0.1;
    }

    if keys.pressed(KeyCode::PageDown) {
        config.range -= 0.1;
    }
}

fn pipe_noop_option<T>(_: In<Option<T>>) {}

fn setup(
    asset_server: Res<AssetServer>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
) {
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
