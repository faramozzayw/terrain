use bevy::prelude::*;
use rayon::{
    iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

use crate::{utils::get_mut_position_from_mesh, BrushConfig};

#[derive(Debug, Reflect, Clone, Copy)]
pub enum BrushKind {
    Sculp,
    Flatten,
    Smooth,
}

#[inline]
pub fn apply_brush(center: Vec2, chunk_mesh: &mut Mesh, config: &BrushConfig) -> Option<()> {
    match config.kind {
        BrushKind::Sculp => apply_sculp_brush(center, chunk_mesh, config),
        BrushKind::Flatten => apply_flatten_brush(center, chunk_mesh, config),
        BrushKind::Smooth => apply_smooth_brush(center, chunk_mesh, config),
    }
}

fn filter_mesh_position(
    position: &mut [f32; 3],
    center: Vec2,
    radius_squared: f32,
    inv_range_squared: f32,
    falloff_exponent: f32,
) -> Option<(&mut [f32; 3], f32)> {
    let this = Vec2::new(position[0], position[2]);
    let distance_squared = center.distance_squared(this);

    if distance_squared <= radius_squared {
        let normalized_distance_squared = (distance_squared * inv_range_squared).min(1.0); // Normalize to [0, 1] squared
        let falloff = (1.0 - normalized_distance_squared).powf(falloff_exponent * 0.5);

        Some((position, falloff))
    } else {
        None
    }
}

fn apply_sculp_brush(center: Vec2, chunk_mesh: &mut Mesh, config: &BrushConfig) -> Option<()> {
    let falloff_exponent = config.falloff_exponent();
    let radius_squared = config.range_squared();
    let inv_range_squared = 1.0 / radius_squared;

    let now = std::time::Instant::now();

    get_mut_position_from_mesh(chunk_mesh)?
        .par_chunks_mut(500)
        .for_each(|chunk| {
            // TODO: merge iterators
            chunk
                .iter_mut()
                .filter_map(|position| {
                    filter_mesh_position(
                        position,
                        center,
                        radius_squared,
                        inv_range_squared,
                        falloff_exponent,
                    )
                })
                .for_each(|(position, falloff)| {
                    position[1] += config.strength * falloff;
                });
        });

    debug!("Sculp duration: {:?}", now.elapsed());

    Some(())
}

fn apply_flatten_brush(center: Vec2, chunk_mesh: &mut Mesh, config: &BrushConfig) -> Option<()> {
    let falloff_exponent = config.falloff_exponent();
    let radius_squared = config.range_squared();
    let inv_range_squared = 1.0 / radius_squared;

    let now = std::time::Instant::now();

    get_mut_position_from_mesh(chunk_mesh)?
        .par_chunks_mut(500)
        .for_each(|chunk| {
            // TODO: merge iterators
            chunk
                .iter_mut()
                .filter_map(|position| {
                    filter_mesh_position(
                        position,
                        center,
                        radius_squared,
                        inv_range_squared,
                        falloff_exponent,
                    )
                })
                .for_each(|(position, falloff)| {
                    let target_y = config.flatten_level.unwrap();
                    position[1] = position[1] * (1.0 - falloff) + target_y * falloff;
                });
        });

    debug!("Flatten duration: {:?}", now.elapsed());

    Some(())
}

const NEIGHBOR_OFFSETS: [(i32, i32); 9] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

// FIXME: multichunk
fn apply_smooth_brush(center: Vec2, chunk_mesh: &mut Mesh, config: &BrushConfig) -> Option<()> {
    let positions = get_mut_position_from_mesh(chunk_mesh)?;
    let falloff_exponent = config.falloff_exponent();
    let radius_squared = config.range_squared();
    let inv_range_squared = 1.0 / radius_squared;
    let inv_range = 1.0 / config.range;

    let Vec2 { x: min_x, y: min_y } = center - config.range;
    let Vec2 { x: max_x, y: max_y } = center + config.range;

    let grid = positions
        .par_iter()
        .map(|pos| {
            let cell = (
                (pos[0] * inv_range).floor() as usize,
                (pos[2] * inv_range).floor() as usize,
            );
            (cell, *pos)
        })
        .fold(std::collections::HashMap::new, |mut acc, (cell, pos)| {
            #[allow(clippy::unwrap_or_default)]
            acc.entry(cell).or_insert_with(Vec::new).push(pos);
            acc
        })
        .reduce(std::collections::HashMap::new, |mut acc1, acc2| {
            for (key, mut value) in acc2 {
                acc1.entry(key).or_default().append(&mut value);
            }
            acc1
        });

    let now = std::time::Instant::now();

    positions
        .par_iter_mut()
        .filter_map(|position| {
            filter_mesh_position(
                position,
                center,
                radius_squared,
                inv_range_squared,
                falloff_exponent,
            )
        })
        .for_each(|(position, falloff)| {
            let pos = Vec2::new(position[0], position[2]);
            let cell_x = (position[0] * inv_range).floor() as i32;
            let cell_y = (position[2] * inv_range).floor() as i32;

            let (sum_heights, count) = NEIGHBOR_OFFSETS
                .par_iter()
                .map(|(dx, dy)| {
                    let mut sum = 0.0;
                    let mut count = 0;

                    let x = (cell_x + dx) as usize;
                    let y = (cell_y + dy) as usize;

                    let Some(neighbors) = grid.get(&(x, y)) else {
                        return (sum, count);
                    };

                    for neighbor in neighbors {
                        let x = neighbor[0];
                        let y = neighbor[2];
                        let neighbor_pos = Vec2::new(x, y);

                        let in_valid_range = x >= min_x && x <= max_x && y >= min_y && y <= max_y;

                        if in_valid_range && pos.distance_squared(neighbor_pos) <= radius_squared {
                            sum += neighbor[1];
                            count += 1;
                        }
                    }

                    (sum, count)
                })
                .reduce(
                    || (0.0, 0),
                    |(sum1, count1), (sum2, count2)| (sum1 + sum2, count1 + count2),
                );

            if count == 0 {
                return;
            }

            let smoothed_height = sum_heights / count as f32;
            let blend_factor = config.strength * falloff;

            position[1] = position[1] * (1.0 - blend_factor) + smoothed_height * blend_factor;
        });

    debug!("Smoth duration: {:?}", now.elapsed());

    Some(())
}
