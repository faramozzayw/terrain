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

struct TerrainMaterial {
  quantize_steps: u32,
}

@group(2) @binding(100)
var<uniform> my_extended_material: TerrainMaterial;

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
  var pbr_input = pbr_input_from_standard_material(in, is_front);

  var color = vec4(in.uv.x, in.uv.y, 1.0, 1.0);

  pbr_input.material.base_color = color;
  pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

#ifdef PREPASS_PIPELINE
  let out = deferred_output(in, pbr_input);
#else
  var out: FragmentOutput;
  out.color = apply_pbr_lighting(pbr_input);

  let quantize_steps = 1000;

  // out.color = vec4<f32>(vec4<u32>(out.color * f32(quantize_steps))) / f32(quantize_steps);
  out.color = main_pass_post_lighting_processing(pbr_input, out.color);
#endif
  
  return out;
}
