# lantern_01: the Blackout Search work lamp

Source: Poly Haven, model **Lantern 01** (`Lantern_01`), by Rajil Jose Macatangay, published 2020-05-28,
https://polyhaven.com/a/Lantern_01 — licence **CC0 1.0** (public domain dedication; no attribution required,
recorded here for provenance).

Downloaded 2026-09-24 as the 1k glTF set (`https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/Lantern_01/`)
and converted to Wavefront OBJ, one file per material primitive, with a small script kept in the build job's
scratch (`gltf_to_obj.py`): glTF y-up to MuJoCo z-up (`x, y, z -> x, -z, y`), texture v flipped, positions,
normals and texture coordinates kept, no decimation. The normal, roughness and metalness maps of the source
are not used (MuJoCo's renderer has no use for them); only the base-colour texture is kept, as PNG because
MuJoCo reads no JPEG.

| file | what | sha256 (first 16) |
|---|---|---|
| `lantern_01_brass.obj` | the metal body, cage and bail; 18954 vertices, 30830 faces; textured | 2952c916ac2eb84f |
| `lantern_01_glass.obj` | the glass globe; 1716 vertices, 3072 faces; rendered translucent and emissive | 0dd095e86b7069bd |
| `lantern_01_brass_diff_1k.png` | the base-colour texture, 1024 x 1024 | 6151e7ad285049d6 |

Source extents at scale 1: 0.122 x 0.097 x 0.294 m (the bail apex at z 0.294, the glass from z 0.061 to
0.130, radius 0.035). The task builds it at `blackout/lamp.py`'s `MESH_SCALE`. The meshes draw only; the
collision model is the primitive stack in `blackout/lamp.py` (foot, body, cap, the bail apex as a thickened
bar the gripper pinches).
