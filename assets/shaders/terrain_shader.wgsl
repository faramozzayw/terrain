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
  pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing, calculate_tbn_mikktspace, apply_normal_mapping, prepare_world_normal, calculate_view},
  pbr_types::{STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT, PbrInput, pbr_input_new},
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
var array_normal: texture_2d_array<f32>;
@group(2) @binding(105) 
var array_normal_sampler: sampler;

@group(2) @binding(106)
var<uniform> layers: u32;

@group(2) @binding(107)
var<uniform> tiling_factor: f32;

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
  let height = in.world_position.y;
  var pbr_input = pbr_input_from_standard_material(in, is_front);

  let max_grass_level = 5.0;
  let max_rock_level = 10.0;

  let tiled_uv = fract(in.uv * 5.0);

  let sand = textureSample(array_texture, array_texture_sampler, tiled_uv, 1);
  let sand_normal = textureSample(array_normal, array_normal_sampler, tiled_uv, 1).xyz;

  let dirt = textureSample(array_texture, array_texture_sampler, tiled_uv, 2);
  let dirt_normal = textureSample(array_normal, array_normal_sampler, tiled_uv, 2).xyz;

  let grass = textureSample(array_texture, array_texture_sampler,tiled_uv, 3);
  let grass_normal = textureSample(array_normal, array_normal_sampler, tiled_uv, 3).xyz;

  let rock = textureSample(array_texture, array_texture_sampler, tiled_uv, 4);
  let rock_normal = textureSample(array_normal, array_normal_sampler, tiled_uv, 4).xyz;

  let snow = textureSample(array_texture, array_texture_sampler, tiled_uv, 5);
  let snow_normal = textureSample(array_normal, array_normal_sampler, tiled_uv, 5).xyz;

  var color = sand;
  var normal_from_map = sand_normal;

  color = mix(sand, dirt, smoothstep(-20.0, -10.0, height));
  normal_from_map = mix(sand_normal, dirt_normal, smoothstep(-20.0, -10.0, height));

  color = mix(color, grass, smoothstep(0.0, max_grass_level, height));
  normal_from_map = mix(normal_from_map, dirt_normal, smoothstep(0.0, max_grass_level, height));

  color = mix(color, rock, smoothstep(max_grass_level, max_rock_level, height));
  normal_from_map = mix(normal_from_map, rock_normal, smoothstep(max_grass_level, max_rock_level, height));

  color = mix(color, snow, smoothstep(max_rock_level + 5.0, 50.0, height));
  normal_from_map = mix(normal_from_map, snow_normal, smoothstep(max_rock_level + 5.0, 50.0, height));

  let material_index = textureLoad(
    material_index_map,
    vec2<i32>(floor(in.uv * vec2<f32>(textureDimensions(material_index_map)))),
    0
  ).r;

  let use_material_texture = material_index > 0;
  color = select(color, textureSample(array_texture, array_texture_sampler, tiled_uv, material_index), use_material_texture);
  normal_from_map = select(vec4(normal_from_map, 1.0), textureSample(array_normal, array_normal_sampler, tiled_uv, material_index), use_material_texture).rgb;

  let use_default_texture = material_index >= layers;
  color = select(color, textureSample(array_texture, array_texture_sampler, tiled_uv, 0), use_default_texture);
  normal_from_map = select(vec4(normal_from_map, 1.0), textureSample(array_normal, array_normal_sampler, tiled_uv, 0), use_default_texture).rgb;

  pbr_input.world_normal = prepare_world_normal(
     in.world_normal,
      false,
      is_front,
  );
  pbr_input.N = normalize(pbr_input.world_normal);

  #ifdef VERTEX_TANGENTS
    let TBN = calculate_tbn_mikktspace(in.world_normal, in.world_tangent);

    pbr_input.world_normal = normalize(TBN * normal_from_map);
    pbr_input.N = apply_normal_mapping(
        pbr_input.material.flags,
        TBN,
        false,
        is_front,
        pbr_input.world_normal.rgb,
    );
  #endif

  pbr_input.V = calculate_view(in.world_position, pbr_input.is_orthographic);
  pbr_input.material.base_color = color;

#ifdef PREPASS_PIPELINE
  let out = deferred_output(in, pbr_input);
#else
  var out: FragmentOutput;
  out.color = apply_pbr_lighting(pbr_input);

  out.color = main_pass_post_lighting_processing(pbr_input, out.color);
#endif
 
  return out;
}
