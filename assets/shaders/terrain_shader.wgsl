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
var array_texture: texture_2d_array<f32>;
@group(2) @binding(101) 
var array_texture_sampler: sampler;

@group(2) @binding(102)
var material_index_map: texture_2d<u32>;
@group(2) @binding(103)
var material_index_sampler: sampler;

@group(2) @binding(104)
var<uniform> layers: u32;

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
  let height = in.world_position.y;
  var pbr_input = pbr_input_from_standard_material(in, is_front);

  let max_grass_level = 5.0;
  let max_rock_level = 10.0;

  let sand = textureSample(array_texture, array_texture_sampler, in.uv, 1);
  let dirt = textureSample(array_texture, array_texture_sampler, in.uv, 2);
  let grass = textureSample(array_texture, array_texture_sampler, in.uv, 3);
  let rock = textureSample(array_texture, array_texture_sampler, in.uv, 4);
  let snow = textureSample(array_texture, array_texture_sampler, in.uv, 5);

  var color = sand;

  color = mix(sand, dirt, smoothstep(-20.0, -10.0, height));
  color = mix(color, grass, smoothstep(0.0, max_grass_level, height));
  color = mix(color, rock, smoothstep(max_grass_level, max_rock_level, height));
  color = mix(color, snow, smoothstep(max_rock_level + 5.0, 50.0, height));

  let material_index = textureLoad(
    material_index_map,
    vec2<i32>(floor(in.uv * vec2<f32>(textureDimensions(material_index_map)))),
    0
  ).r;

  let use_material_texture = material_index > 0;
  color = select(color, textureSample(array_texture, array_texture_sampler, in.uv, material_index), use_material_texture);

  let use_default_texture = material_index >= layers;
  color = select(color, textureSample(array_texture, array_texture_sampler, in.uv, 0), use_default_texture);

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
