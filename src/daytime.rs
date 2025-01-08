use super::time::TimeTracker;
use bevy::{pbr::light_consts::lux::AMBIENT_DAYLIGHT, prelude::*};
use bevy_atmosphere::prelude::*;

#[derive(Component)]
struct Sun;

impl Sun {
    fn time_to_angle(time_secs: f32) -> f32 {
        time_secs * 360.0 / 86_400.0
    }

    /// angle in degrees
    fn illuminance(angle: f32) -> f32 {
        let rad = angle.to_radians() / 2.0;

        rad.sin().max(0.0).powf(2.0) * AMBIENT_DAYLIGHT
    }

    /// angle in degrees
    fn light_rotation(angle: f32) -> Quat {
        Quat::from_rotation_x(-(angle.to_radians() / 2.0))
    }

    /// angle in degrees
    fn position(angle: f32) -> Vec3 {
        let t = angle.to_radians() / 2.0;

        Vec3::new(0., t.sin(), t.cos())
    }
}

fn daylight_cycle(
    mut atmosphere: AtmosphereMut<Nishita>,
    mut query: Query<(&mut Transform, &mut DirectionalLight), With<Sun>>,
    game_time: Res<TimeTracker>,
) {
    if !game_time.enabled {
        return;
    }
    let angle = Sun::time_to_angle(game_time.get_game_day_time_in_secs());
    atmosphere.sun_position = Sun::position(angle);

    let Some((mut light_trans, mut light)) = query.single_mut().into() else {
        return;
    };

    light_trans.rotation = Sun::light_rotation(angle);
    light.illuminance = Sun::illuminance(angle);
}

fn setup_sun(mut commands: Commands) {
    commands.spawn((
        Name::new("Sun"),
        Sun,
        DirectionalLight {
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(5.0, 5.0, 5.0),
    ));
}

pub struct DaylightCyclePlugin;

impl Plugin for DaylightCyclePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(AtmospherePlugin)
            .insert_resource(AtmosphereModel::default())
            .add_systems(Startup, setup_sun)
            .add_systems(Update, daylight_cycle);
    }
}
