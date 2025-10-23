"""
Scene management system for 3D rendering
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from OpenGL.GL import *


class MaterialType(Enum):
    BASIC = "basic"
    PHONG = "phong"
    PBR = "pbr"


@dataclass
class Material:
    """Material properties for rendering"""
    name: str
    material_type: MaterialType = MaterialType.PHONG
    diffuse_color: np.ndarray = None
    specular_color: np.ndarray = None
    ambient_color: np.ndarray = None
    shininess: float = 32.0
    transparency: float = 1.0
    texture_id: Optional[int] = None
    normal_map_id: Optional[int] = None
    
    def __post_init__(self):
        if self.diffuse_color is None:
            self.diffuse_color = np.array([0.8, 0.8, 0.8], dtype=np.float32)
        if self.specular_color is None:
            self.specular_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        if self.ambient_color is None:
            self.ambient_color = np.array([0.2, 0.2, 0.2], dtype=np.float32)


@dataclass
class Mesh:
    """3D mesh data"""
    vertices: np.ndarray
    indices: np.ndarray
    normals: np.ndarray = None
    tex_coords: np.ndarray = None
    vao: int = 0
    vbo: int = 0
    ebo: int = 0
    
    def __post_init__(self):
        if self.normals is None:
            self.calculate_normals()
        if self.tex_coords is None:
            self.generate_tex_coords()
    
    def calculate_normals(self):
        """Calculate vertex normals"""
        if self.indices is not None and len(self.indices) >= 3:
            normals = np.zeros_like(self.vertices)
            
            # Calculate face normals and accumulate (only for triangles)
            for i in range(0, len(self.indices) - 2, 3):
                if i + 2 < len(self.indices):
                    i0, i1, i2 = self.indices[i], self.indices[i+1], self.indices[i+2]
                    
                    # Ensure indices are valid
                    if i0 < len(self.vertices) and i1 < len(self.vertices) and i2 < len(self.vertices):
                        v0, v1, v2 = self.vertices[i0], self.vertices[i1], self.vertices[i2]
                        
                        # Calculate face normal
                        edge1 = v1 - v0
                        edge2 = v2 - v0
                        face_normal = np.cross(edge1, edge2)
                        
                        # Normalize face normal (handle zero-length normals)
                        norm = np.linalg.norm(face_normal)
                        if norm > 1e-8:
                            face_normal = face_normal / norm
                            
                            # Accumulate to vertex normals
                            normals[i0] += face_normal
                            normals[i1] += face_normal
                            normals[i2] += face_normal
            
            # Normalize vertex normals
            for i in range(len(normals)):
                norm = np.linalg.norm(normals[i])
                if norm > 1e-8:
                    normals[i] /= norm
                else:
                    # Default normal pointing up for degenerate cases
                    normals[i] = np.array([0.0, 1.0, 0.0])
            
            self.normals = normals
        else:
            # Default normals for non-triangle meshes or empty indices
            self.normals = np.tile([0.0, 1.0, 0.0], (len(self.vertices), 1)).astype(np.float32)
    
    def generate_tex_coords(self):
        """Generate basic texture coordinates"""
        # Simple planar mapping
        self.tex_coords = np.column_stack([
            (self.vertices[:, 0] + 1.0) * 0.5,
            (self.vertices[:, 2] + 1.0) * 0.5
        ]).astype(np.float32)


@dataclass
class RenderableObject:
    """Object that can be rendered in the scene"""
    name: str
    mesh: Mesh
    material: Material
    transform: np.ndarray = None
    visible: bool = True
    cast_shadows: bool = True
    receive_shadows: bool = True
    
    def __post_init__(self):
        if self.transform is None:
            self.transform = np.eye(4, dtype=np.float32)


class SceneManager:
    """Manages the 3D scene and renderable objects"""
    
    def __init__(self):
        self.objects: Dict[str, RenderableObject] = {}
        self.materials: Dict[str, Material] = {}
        self.meshes: Dict[str, Mesh] = {}
        self._setup_default_materials()
        self._setup_primitive_meshes()
    
    def _setup_default_materials(self):
        """Create default materials"""
        # Default material
        default_material = Material(
            name="default",
            diffuse_color=np.array([0.8, 0.8, 0.8], dtype=np.float32),
            specular_color=np.array([0.5, 0.5, 0.5], dtype=np.float32),
            shininess=32.0
        )
        self.materials["default"] = default_material
        
        # Vehicle materials
        vehicle_material = Material(
            name="vehicle",
            diffuse_color=np.array([0.2, 0.4, 0.8], dtype=np.float32),
            specular_color=np.array([0.8, 0.8, 0.8], dtype=np.float32),
            shininess=64.0
        )
        self.materials["vehicle"] = vehicle_material
        
        # Ground material
        ground_material = Material(
            name="ground",
            diffuse_color=np.array([0.4, 0.4, 0.4], dtype=np.float32),
            specular_color=np.array([0.1, 0.1, 0.1], dtype=np.float32),
            shininess=8.0
        )
        self.materials["ground"] = ground_material
        
        # Grid material
        grid_material = Material(
            name="grid",
            diffuse_color=np.array([0.3, 0.3, 0.3], dtype=np.float32),
            specular_color=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            shininess=1.0,
            transparency=0.5
        )
        self.materials["grid"] = grid_material
    
    def _setup_primitive_meshes(self):
        """Create basic primitive meshes"""
        # Cube mesh
        cube_vertices = np.array([
            # Front face
            -1.0, -1.0,  1.0,
             1.0, -1.0,  1.0,
             1.0,  1.0,  1.0,
            -1.0,  1.0,  1.0,
            # Back face
            -1.0, -1.0, -1.0,
            -1.0,  1.0, -1.0,
             1.0,  1.0, -1.0,
             1.0, -1.0, -1.0,
        ], dtype=np.float32).reshape(-1, 3)
        
        cube_indices = np.array([
            0, 1, 2, 2, 3, 0,  # Front
            4, 5, 6, 6, 7, 4,  # Back
            5, 3, 2, 2, 6, 5,  # Top
            4, 7, 1, 1, 0, 4,  # Bottom
            3, 5, 4, 4, 0, 3,  # Left
            1, 7, 6, 6, 2, 1   # Right
        ], dtype=np.uint32)
        
        self.meshes["cube"] = Mesh(vertices=cube_vertices, indices=cube_indices)
        
        # Plane mesh (for ground)
        plane_vertices = np.array([
            -1.0, 0.0, -1.0,
             1.0, 0.0, -1.0,
             1.0, 0.0,  1.0,
            -1.0, 0.0,  1.0,
        ], dtype=np.float32).reshape(-1, 3)
        
        plane_indices = np.array([
            0, 1, 2, 2, 3, 0
        ], dtype=np.uint32)
        
        self.meshes["plane"] = Mesh(vertices=plane_vertices, indices=plane_indices)
        
        # Grid mesh (for line rendering, no triangles)
        grid_size = 50
        grid_vertices = []
        
        # Create grid lines
        for i in range(-grid_size, grid_size + 1):
            # Horizontal lines
            grid_vertices.extend([
                [-grid_size, 0.0, i],
                [grid_size, 0.0, i]
            ])
            # Vertical lines
            grid_vertices.extend([
                [i, 0.0, -grid_size],
                [i, 0.0, grid_size]
            ])
        
        grid_vertices = np.array(grid_vertices, dtype=np.float32)
        # Grid uses line indices, not triangles
        grid_indices = np.arange(len(grid_vertices), dtype=np.uint32)
        
        # Create grid mesh without normal calculation (lines don't need normals)
        grid_mesh = Mesh(vertices=grid_vertices, indices=grid_indices)
        grid_mesh.normals = np.zeros_like(grid_vertices)  # Set empty normals for lines
        self.meshes["grid"] = grid_mesh
    
    def add_object(self, obj: RenderableObject):
        """Add an object to the scene"""
        self.objects[obj.name] = obj
        self._setup_object_buffers(obj)
    
    def remove_object(self, name: str):
        """Remove an object from the scene"""
        if name in self.objects:
            obj = self.objects[name]
            self._cleanup_object_buffers(obj)
            del self.objects[name]
    
    def get_object(self, name: str) -> Optional[RenderableObject]:
        """Get an object by name"""
        return self.objects.get(name)
    
    def add_material(self, material: Material):
        """Add a material to the library"""
        self.materials[material.name] = material
    
    def get_material(self, name: str) -> Optional[Material]:
        """Get a material by name"""
        return self.materials.get(name, self.materials["default"])
    
    def add_mesh(self, name: str, mesh: Mesh):
        """Add a mesh to the library"""
        self.meshes[name] = mesh
    
    def get_mesh(self, name: str) -> Optional[Mesh]:
        """Get a mesh by name"""
        return self.meshes.get(name)
    
    def _setup_object_buffers(self, obj: RenderableObject):
        """Setup OpenGL buffers for an object"""
        mesh = obj.mesh
        
        # Generate buffers
        mesh.vao = glGenVertexArrays(1)
        mesh.vbo = glGenBuffers(1)
        mesh.ebo = glGenBuffers(1)
        
        glBindVertexArray(mesh.vao)
        
        # Vertex buffer
        glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo)
        
        # Combine vertex data
        vertex_data = mesh.vertices
        if mesh.normals is not None:
            vertex_data = np.column_stack([vertex_data, mesh.normals])
        if mesh.tex_coords is not None:
            vertex_data = np.column_stack([vertex_data, mesh.tex_coords])
        
        glBufferData(GL_ARRAY_BUFFER, vertex_data.astype(np.float32), GL_STATIC_DRAW)
        
        # Element buffer
        if mesh.indices is not None:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices, GL_STATIC_DRAW)
        
        # Vertex attributes
        stride = 3 * 4  # 3 floats for position
        
        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Normal attribute
        if mesh.normals is not None:
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * 4))
            glEnableVertexAttribArray(1)
            stride += 3 * 4
        
        # Texture coordinate attribute
        if mesh.tex_coords is not None:
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6 * 4))
            glEnableVertexAttribArray(2)
        
        glBindVertexArray(0)
    
    def _cleanup_object_buffers(self, obj: RenderableObject):
        """Clean up OpenGL buffers for an object"""
        mesh = obj.mesh
        if mesh.vao:
            glDeleteVertexArrays(1, [mesh.vao])
        if mesh.vbo:
            glDeleteBuffers(1, [mesh.vbo])
        if mesh.ebo:
            glDeleteBuffers(1, [mesh.ebo])
    
    def get_visible_objects(self) -> List[RenderableObject]:
        """Get all visible objects in the scene"""
        return [obj for obj in self.objects.values() if obj.visible]
    
    def update(self, delta_time: float):
        """Update scene objects"""
        # Update any animated objects or effects
        pass
    
    def cleanup(self):
        """Clean up all scene resources"""
        for obj in self.objects.values():
            self._cleanup_object_buffers(obj)
        self.objects.clear()
        self.materials.clear()
        self.meshes.clear()