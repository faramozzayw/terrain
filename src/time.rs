use bevy::prelude::*;

pub struct TimeTrackerPlugin {
    timescale: f32,
}

impl TimeTrackerPlugin {
    pub fn new(timescale: f32) -> Self {
        Self { timescale }
    }
}

impl Plugin for TimeTrackerPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<TimeTracker>()
            .insert_resource(TimeTracker::new(self.timescale))
            .add_systems(Update, tick_game_time);
    }
}

#[derive(Resource, Clone, Default, Reflect)]
#[reflect(Resource)]
pub struct TimeTracker {
    pub game_seconds: f32,
    timescale: f32,
    pub enabled: bool,
}

impl TimeTracker {
    pub fn new(timescale: f32) -> Self {
        Self {
            game_seconds: 0.0,
            timescale,
            enabled: true,
        }
    }

    /* all in secs */
    pub const MINUTE: f32 = 60.0;
    pub const HOUR: f32 = Self::MINUTE * 60.0;
    pub const DAY: f32 = Self::HOUR * 24.0;

    pub fn advance_by(&mut self, real_delta_seconds: f32) {
        self.game_seconds += real_delta_seconds * self.timescale;
    }

    #[inline]
    pub fn get_game_day_time_in_secs(&self) -> f32 {
        self.game_seconds % Self::DAY
    }
}

fn tick_game_time(mut game_time: ResMut<TimeTracker>, time: Res<Time>) {
    if !game_time.enabled {
        return;
    }

    game_time.advance_by(time.delta_secs());
}
