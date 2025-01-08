BLACK_TEXTURE=./assets/textures/sources/BlackColor.png
TEXTURE_ARRAY_SOURCE=./assets/textures/sources

TEXTURE_ARRAY_OUTPUT=./assets/textures/array_texture.png
NORMAL_ARRAY_OUTPUT=./assets/textures/array_normal.png
ROUGHNESS_ARRAY_OUTPUT=./assets/textures/array_roughness.png
DISPLACEMENT_ARRAY_OUTPUT=./assets/textures/array_displacement.png

NORMAL_TYPE=DX

TEXTURE_SOURCES=\
	$(TEXTURE_ARRAY_SOURCE)/Ground080_1K-JPG/Ground080_1K-JPG \
	$(TEXTURE_ARRAY_SOURCE)/Ground082L_1K-JPG/Ground082L_1K-JPG \
	$(TEXTURE_ARRAY_SOURCE)/Grass001_1K-JPG/Grass001_1K-JPG \
	$(TEXTURE_ARRAY_SOURCE)/Rock030_1K-JPG/Rock030_1K-JPG \
	$(TEXTURE_ARRAY_SOURCE)/Snow010A_1K-JPG/Snow010A_1K-JPG

# Function to generate array
define GENERATE_ARRAY
	convert -append \
		$(BLACK_TEXTURE) \
		$(foreach src,$(TEXTURE_SOURCES),$(src)_$(1)) \
		$(2)
endef

# Targets
generate-texture-array:
	$(call GENERATE_ARRAY,Color.jpg,$(TEXTURE_ARRAY_OUTPUT))

generate-normal-array:
	$(call GENERATE_ARRAY,Normal$(NORMAL_TYPE).jpg,$(NORMAL_ARRAY_OUTPUT))

generate-roughness-array:
	$(call GENERATE_ARRAY,Roughness.jpg,$(ROUGHNESS_ARRAY_OUTPUT))

generate-displacement-array:
	$(call GENERATE_ARRAY,Displacement.jpg,$(DISPLACEMENT_ARRAY_OUTPUT))

