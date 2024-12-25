#import bevy_pbr::{
  pbr_fragment::pbr_input_from_standard_material,
  pbr_functions::alpha_discard,
}

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
  prepass_io::{VertexOutput, FragmentOutput},
  pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
  forward_io::{VertexOutput, FragmentOutput},
  pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif

@group(2) @binding(100)
var grass_texture: texture_2d<f32>;
@group(2) @binding(101)
var grass_sampler: sampler;

@group(2) @binding(102)
var rock_texture: texture_2d<f32>;
@group(2) @binding(103)
var rock_sampler: sampler;

@group(2) @binding(104)
var snow_texture: texture_2d<f32>;
@group(2) @binding(105)
var snow_sampler: sampler;

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
  let height = in.world_position.y;
  var pbr_input = pbr_input_from_standard_material(in, is_front);

  let max_grass_level = 5.0;
  let max_rock_level = 10.0;
  let transtion_width = 3.0;

  let grass = textureSample(grass_texture, grass_sampler, in.uv);
  let rock = textureSample(rock_texture, rock_sampler, in.uv);
  let snow = textureSample(snow_texture, snow_sampler, in.uv);

  var color = mix(grass, rock, smoothstep(max_grass_level, max_rock_level, height));
  color = mix(color, snow, smoothstep(max_rock_level + 10.0, 40.0, height));

  pbr_input.material.base_color = color;
  pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

#ifdef PREPASS_PIPELINE
  let out = deferred_output(in, pbr_input);
#else
  var out: FragmentOutput;
  out.color = apply_pbr_lighting(pbr_input);

  out.color = main_pass_post_lighting_processing(pbr_input, out.color);
#endif
 
  return out;
}
