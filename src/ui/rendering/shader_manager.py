"""
Shader management system for OpenGL rendering
"""

import os
from typing import Dict, Optional
from OpenGL.GL import *
import numpy as np


class ShaderManager:
    """Manages OpenGL shaders and shader programs"""
    
    def __init__(self):
        self.shaders: Dict[str, int] = {}
        self.programs: Dict[str, int] = {}
        self._default_shaders_created = False
    
    def create_shader(self, shader_type: int, source: str) -> int:
        """Create and compile a shader"""
        try:
            shader = glCreateShader(shader_type)
            glShaderSource(shader, source)
            glCompileShader(shader)
            
            # Check compilation status
            if not glGetShaderiv(shader, GL_COMPILE_STATUS):
                error = glGetShaderInfoLog(shader).decode()
                glDeleteShader(shader)
                raise RuntimeError(f"Shader compilation failed: {error}")
            
            return shader
        except NameError:
            # OpenGL not available (test environment)
            return 0
    
    def create_program(self, name: str, vertex_source: str, fragment_source: str) -> int:
        """Create a shader program from vertex and fragment shaders"""
        try:
            vertex_shader = self.create_shader(GL_VERTEX_SHADER, vertex_source)
            fragment_shader = self.create_shader(GL_FRAGMENT_SHADER, fragment_source)
            
            program = glCreateProgram()
            glAttachShader(program, vertex_shader)
            glAttachShader(program, fragment_shader)
            glLinkProgram(program)
            
            # Check linking status
            if not glGetProgramiv(program, GL_LINK_STATUS):
                error = glGetProgramInfoLog(program).decode()
                glDeleteProgram(program)
                raise RuntimeError(f"Program linking failed: {error}")
            
            # Clean up shaders
            glDeleteShader(vertex_shader)
            glDeleteShader(fragment_shader)
            
            self.programs[name] = program
            return program
        except NameError:
            # OpenGL not available (test environment)
            self.programs[name] = 0
            return 0
    
    def get_program(self, name: str) -> Optional[int]:
        """Get a shader program by name"""
        return self.programs.get(name)
    
    def use_program(self, name: str):
        """Use a shader program"""
        program = self.get_program(name)
        if program:
            glUseProgram(program)
        else:
            raise ValueError(f"Shader program '{name}' not found")
    
    def set_uniform_matrix4(self, program: int, name: str, matrix: np.ndarray):
        """Set a 4x4 matrix uniform"""
        location = glGetUniformLocation(program, name)
        if location != -1:
            glUniformMatrix4fv(location, 1, GL_FALSE, matrix.astype(np.float32))
    
    def set_uniform_vector3(self, program: int, name: str, vector: np.ndarray):
        """Set a 3D vector uniform"""
        location = glGetUniformLocation(program, name)
        if location != -1:
            glUniform3fv(location, 1, vector.astype(np.float32))
    
    def set_uniform_float(self, program: int, name: str, value: float):
        """Set a float uniform"""
        location = glGetUniformLocation(program, name)
        if location != -1:
            glUniform1f(location, value)
    
    def create_default_shaders(self):
        """Create default shaders for basic rendering"""
        if self._default_shaders_created:
            return
        
        # Basic vertex shader
        basic_vertex = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aNormal;
        layout (location = 2) in vec2 aTexCoord;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform mat3 normalMatrix;
        
        out vec3 FragPos;
        out vec3 Normal;
        out vec2 TexCoord;
        
        void main() {
            FragPos = vec3(model * vec4(aPos, 1.0));
            Normal = normalMatrix * aNormal;
            TexCoord = aTexCoord;
            
            gl_Position = projection * view * vec4(FragPos, 1.0);
        }
        """
        
        # Basic fragment shader with Phong lighting
        basic_fragment = """
        #version 330 core
        out vec4 FragColor;
        
        in vec3 FragPos;
        in vec3 Normal;
        in vec2 TexCoord;
        
        uniform vec3 lightPos;
        uniform vec3 lightColor;
        uniform vec3 viewPos;
        uniform vec3 objectColor;
        uniform float ambientStrength;
        uniform float specularStrength;
        uniform int shininess;
        
        void main() {
            // Ambient
            vec3 ambient = ambientStrength * lightColor;
            
            // Diffuse
            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(lightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;
            
            // Specular
            vec3 viewDir = normalize(viewPos - FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
            vec3 specular = specularStrength * spec * lightColor;
            
            vec3 result = (ambient + diffuse + specular) * objectColor;
            FragColor = vec4(result, 1.0);
        }
        """
        
        # Grid shader for ground plane
        grid_vertex = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        
        out vec3 WorldPos;
        
        void main() {
            WorldPos = vec3(model * vec4(aPos, 1.0));
            gl_Position = projection * view * vec4(WorldPos, 1.0);
        }
        """
        
        grid_fragment = """
        #version 330 core
        out vec4 FragColor;
        
        in vec3 WorldPos;
        
        uniform float gridSize;
        uniform vec3 gridColor;
        uniform float lineWidth;
        
        void main() {
            vec2 coord = WorldPos.xz / gridSize;
            vec2 grid = abs(fract(coord - 0.5) - 0.5) / fwidth(coord);
            float line = min(grid.x, grid.y);
            float alpha = 1.0 - min(line, 1.0);
            
            if (alpha < 0.1) discard;
            
            FragColor = vec4(gridColor, alpha * 0.5);
        }
        """
        
        # Create shader programs (using OpenGL constants)
        try:
            self.create_program("basic", basic_vertex, basic_fragment)
            self.create_program("grid", grid_vertex, grid_fragment)
        except NameError:
            # OpenGL constants not available (likely in test environment)
            print("OpenGL constants not available - skipping shader creation")
        
        self._default_shaders_created = True
    
    def cleanup(self):
        """Clean up all shader resources"""
        for program in self.programs.values():
            glDeleteProgram(program)
        self.programs.clear()
        self.shaders.clear()