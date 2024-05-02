# Visualization of particles with gravity
# Source: http://enja.org/2010/08/27/adventures-in-opencl-part-2-particles-with-opengl/

import sys

import numpy as np
from OpenGL import GL, GLU, GLUT
from OpenGL.arrays import vbo
from OpenGL.GL import (
    GL_ARRAY_BUFFER, GL_BLEND, GL_COLOR_ARRAY, GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT, GL_DYNAMIC_DRAW, GL_FLOAT, GL_MODELVIEW,
    GL_ONE_MINUS_SRC_ALPHA, GL_POINT_SMOOTH, GL_POINTS, GL_PROJECTION, GL_SRC_ALPHA,
    GL_VERTEX_ARRAY)
from OpenGL.GLUT import GLUT_DEPTH, GLUT_DOUBLE, GLUT_RGBA

import pyopencl as cl
from pyopencl.tools import get_gl_sharing_context_properties


mf = cl.mem_flags

width = 800
height = 600
num_particles = 100000
time_step = 0.005
mouse_down = False
mouse_old = {"x": 0.0, "y": 0.0}
rotate = {"x": 0.0, "y": 0.0, "z": 0.0}
translate = {"x": 0.0, "y": 0.0, "z": 0.0}
initial_translate = {"x": 0.0, "y": 0.0, "z": -2.5}


def glut_window():
    GLUT.glutInit(sys.argv)
    GLUT.glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    GLUT.glutInitWindowSize(width, height)
    GLUT.glutInitWindowPosition(0, 0)
    window = GLUT.glutCreateWindow("Particle Simulation")

    GLUT.glutDisplayFunc(on_display)  # Called by GLUT every frame
    GLUT.glutKeyboardFunc(on_key)
    GLUT.glutMouseFunc(on_click)
    GLUT.glutMotionFunc(on_mouse_move)
    GLUT.glutTimerFunc(10, on_timer, 10)  # Call draw every 30 ms

    GL.glViewport(0, 0, width, height)
    GL.glMatrixMode(GL_PROJECTION)
    GL.glLoadIdentity()
    GLU.gluPerspective(60.0, width / float(height), 0.1, 1000.0)

    return window


def initial_buffers(num_particles):
    rng = np.random.default_rng()

    np_position = np.empty((num_particles, 4), dtype=np.float32)
    np_color = np.empty((num_particles, 4), dtype=np.float32)
    np_velocity = np.empty((num_particles, 4), dtype=np.float32)

    np_position[:, 0] = np.sin(
        np.arange(0.0, num_particles) * 2.001 * np.pi / num_particles
    )
    np_position[:, 0] *= rng.integers(num_particles) / 3.0 + 0.2
    np_position[:, 1] = np.cos(
        np.arange(0.0, num_particles) * 2.001 * np.pi / num_particles
    )
    np_position[:, 1] *= rng.integers(num_particles) / 3.0 + 0.2
    np_position[:, 2] = 0.0
    np_position[:, 3] = 1.0

    np_color[:, :] = [1.0, 1.0, 1.0, 1.0]  # White particles

    np_velocity[:, 0] = np_position[:, 0] * 2.0
    np_velocity[:, 1] = np_position[:, 1] * 2.0
    np_velocity[:, 2] = 3.0
    np_velocity[:, 3] = rng.integers(num_particles)

    gl_position = vbo.VBO(
        data=np_position, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER
    )
    gl_position.bind()
    gl_color = vbo.VBO(data=np_color, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
    gl_color.bind()

    return (np_position, np_velocity, gl_position, gl_color)


def on_timer(t):
    GLUT.glutTimerFunc(t, on_timer, t)
    GLUT.glutPostRedisplay()


def on_key(*args):
    if args[0] == "\033" or args[0] == "q":
        sys.exit()


def on_click(button, state, x, y):
    mouse_old["x"] = x
    mouse_old["y"] = y


def on_mouse_move(x, y):
    rotate["x"] += (y - mouse_old["y"]) * 0.2
    rotate["y"] += (x - mouse_old["x"]) * 0.2

    mouse_old["x"] = x
    mouse_old["y"] = y


def on_display():
    """Render the particles"""
    # Update or particle positions by calling the OpenCL kernel
    cl.enqueue_acquire_gl_objects(queue, [cl_gl_position, cl_gl_color])
    kernelargs = (
        cl_gl_position,
        cl_gl_color,
        cl_velocity,
        cl_start_position,
        cl_start_velocity,
        np.float32(time_step),
    )
    program.particle_fountain(queue, (num_particles,), None, *(kernelargs))
    cl.enqueue_release_gl_objects(queue, [cl_gl_position, cl_gl_color])
    queue.finish()
    GL.glFlush()

    GL.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    GL.glMatrixMode(GL_MODELVIEW)
    GL.glLoadIdentity()

    # Handle mouse transformations
    GL.glTranslatef(initial_translate["x"], initial_translate["y"], initial_translate["z"])
    GL.glRotatef(rotate["x"], 1, 0, 0)
    GL.glRotatef(rotate["y"], 0, 1, 0)  # we switched around the axis so make this rotate_z
    GL.glTranslatef(translate["x"], translate["y"], translate["z"])

    # Render the particles
    GL.glEnable(GL_POINT_SMOOTH)
    GL.glPointSize(2)
    GL.glEnable(GL_BLEND)
    GL.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Set up the VBOs
    gl_color.bind()
    GL.glColorPointer(4, GL_FLOAT, 0, gl_color)
    gl_position.bind()
    GL.glVertexPointer(4, GL_FLOAT, 0, gl_position)
    GL.glEnableClientState(GL_VERTEX_ARRAY)
    GL.glEnableClientState(GL_COLOR_ARRAY)

    # Draw the VBOs
    GL.glDrawArrays(GL_POINTS, 0, num_particles)

    GL.glDisableClientState(GL_COLOR_ARRAY)
    GL.glDisableClientState(GL_VERTEX_ARRAY)

    GL.glDisable(GL_BLEND)

    GLUT.glutSwapBuffers()


window = glut_window()

(np_position, np_velocity, gl_position, gl_color) = initial_buffers(num_particles)

platform = cl.get_platforms()[0]
context = cl.Context(
    properties=[(cl.context_properties.PLATFORM, platform)]
    + get_gl_sharing_context_properties()
)
queue = cl.CommandQueue(context)

cl_velocity = cl.Buffer(context, mf.COPY_HOST_PTR, hostbuf=np_velocity)
cl_start_position = cl.Buffer(
    context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_position
)
cl_start_velocity = cl.Buffer(
    context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_velocity
)

cl_gl_position = cl.GLBuffer(context, mf.READ_WRITE, int(gl_position))
cl_gl_color = cl.GLBuffer(context, mf.READ_WRITE, int(gl_color))

kernel = """__kernel void particle_fountain(__global float4* position,
                                            __global float4* color,
                                            __global float4* velocity,
                                            __global float4* start_position,
                                            __global float4* start_velocity,
                                            float time_step)
{
    unsigned int i = get_global_id(0);
    float4 p = position[i];
    float4 v = velocity[i];
    float life = velocity[i].w;
    life -= time_step;
    if (life <= 0.f)
    {
        p = start_position[i];
        v = start_velocity[i];
        life = 1.0f;
    }

    v.z -= 9.8f*time_step;
    p.x += v.x*time_step;
    p.y += v.y*time_step;
    p.z += v.z*time_step;
    v.w = life;

    position[i] = p;
    velocity[i] = v;

    color[i].w = life; /* Fade points as life decreases */
}"""
program = cl.Program(context, kernel).build()

GLUT.glutMainLoop()
