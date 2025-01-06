BLACK_TEXTURE=./assets/textures/sources/BlackColor.png
TEXTURE_ARRAY_SOURCE=./assets/textures/sources
TEXTURE_ARRAY_OUTPUT=./assets/textures/array_texture.png
NORMAL_ARRAY_OUTPUT=./assets/textures/array_normal.png

NORMAL_TYPE=GL

generate-texture-array:
	 convert -append \
			$(BLACK_TEXTURE) \
			$(TEXTURE_ARRAY_SOURCE)/Ground080_1K-JPG/Ground080_1K-JPG_Color.jpg \
			$(TEXTURE_ARRAY_SOURCE)/Ground082L_1K-JPG/Ground082L_1K-JPG_Color.jpg \
			$(TEXTURE_ARRAY_SOURCE)/Grass001_1K-JPG/Grass001_1K-JPG_Color.jpg \
			$(TEXTURE_ARRAY_SOURCE)/Rock030_1K-JPG/Rock030_1K-JPG_Color.jpg \
			$(TEXTURE_ARRAY_SOURCE)/Snow010A_1K-JPG/Snow010A_1K-JPG_Color.jpg \
			$(TEXTURE_ARRAY_OUTPUT)

generate-normal-array:
	 convert -append \
			$(BLACK_TEXTURE) \
			$(TEXTURE_ARRAY_SOURCE)/Ground080_1K-JPG/Ground080_1K-JPG_Normal$(NORMAL_TYPE).jpg \
			$(TEXTURE_ARRAY_SOURCE)/Ground082L_1K-JPG/Ground082L_1K-JPG_Normal$(NORMAL_TYPE).jpg \
			$(TEXTURE_ARRAY_SOURCE)/Grass001_1K-JPG/Grass001_1K-JPG_Normal$(NORMAL_TYPE).jpg \
			$(TEXTURE_ARRAY_SOURCE)/Rock030_1K-JPG/Rock030_1K-JPG_Normal$(NORMAL_TYPE).jpg \
			$(TEXTURE_ARRAY_SOURCE)/Snow010A_1K-JPG/Snow010A_1K-JPG_Normal$(NORMAL_TYPE).jpg \
			$(NORMAL_ARRAY_OUTPUT)

