# Visualization of particles with gravity
# Source: http://enja.org/2010/08/27/adventures-in-opencl-part-2-particles-with-opengl/

import pyopencl as cl # OpenCL - GPU computing interface
mf = cl.mem_flags
from pyopencl.tools import get_gl_sharing_context_properties
from OpenGL.GL import * # OpenGL - GPU rendering interface
from OpenGL.GLU import * # OpenGL tools (mipmaps, NURBS, perspective projection, shapes)
from OpenGL.GLUT import * # OpenGL tool to make a visualization window
from OpenGL.arrays import vbo 
import numpy # Number tools
import sys # System tools (path, modules, maxint)

width = 800
height = 600
num_particles = 100000
time_step = .005
mouse_down = False
mouse_old = {'x': 0., 'y': 0.}
rotate = {'x': 0., 'y': 0., 'z': 0.}
translate = {'x': 0., 'y': 0., 'z': 0.}
initial_translate = {'x': 0., 'y': 0., 'z': -2.5}

def glut_window():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(width, height)
    glutInitWindowPosition(0, 0)
    window = glutCreateWindow("Particle Simulation")

    glutDisplayFunc(on_display)  # Called by GLUT every frame
    glutKeyboardFunc(on_key)
    glutMouseFunc(on_click)
    glutMotionFunc(on_mouse_move)
    glutTimerFunc(10, on_timer, 10)  # Call draw every 30 ms

    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60., width / float(height), .1, 1000.)

    return(window)

def initial_buffers(num_particles):
    np_position = numpy.ndarray((num_particles, 4), dtype=numpy.float32)
    np_color = numpy.ndarray((num_particles, 4), dtype=numpy.float32)
    np_velocity = numpy.ndarray((num_particles, 4), dtype=numpy.float32)

    np_position[:,0] = numpy.sin(numpy.arange(0., num_particles) * 2.001 * numpy.pi / num_particles) 
    np_position[:,0] *= numpy.random.random_sample((num_particles,)) / 3. + .2
    np_position[:,1] = numpy.cos(numpy.arange(0., num_particles) * 2.001 * numpy.pi / num_particles) 
    np_position[:,1] *= numpy.random.random_sample((num_particles,)) / 3. + .2
    np_position[:,2] = 0.
    np_position[:,3] = 1.

    np_color[:,:] = [1.,1.,1.,1.] # White particles

    np_velocity[:,0] = np_position[:,0] * 2.
    np_velocity[:,1] = np_position[:,1] * 2.
    np_velocity[:,2] = 3.
    np_velocity[:,3] = numpy.random.random_sample((num_particles, ))
    
    gl_position = vbo.VBO(data=np_position, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
    gl_position.bind()
    gl_color = vbo.VBO(data=np_color, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
    gl_color.bind()

    return (np_position, np_velocity, gl_position, gl_color)

def on_timer(t):
    glutTimerFunc(t, on_timer, t)
    glutPostRedisplay()

def on_key(*args):
    if args[0] == '\033' or args[0] == 'q':
        sys.exit()

def on_click(button, state, x, y):
    mouse_old['x'] = x
    mouse_old['y'] = y

def on_mouse_move(x, y):
    rotate['x'] += (y - mouse_old['y']) * .2
    rotate['y'] += (x - mouse_old['x']) * .2

    mouse_old['x'] = x
    mouse_old['y'] = y

def on_display():
    """Render the particles"""        
    # Update or particle positions by calling the OpenCL kernel
    cl.enqueue_acquire_gl_objects(queue, [cl_gl_position, cl_gl_color])
    kernelargs = (cl_gl_position, cl_gl_color, cl_velocity, cl_start_position, cl_start_velocity, numpy.float32(time_step))
    program.particle_fountain(queue, (num_particles,), None, *(kernelargs))
    cl.enqueue_release_gl_objects(queue, [cl_gl_position, cl_gl_color])
    queue.finish()
    glFlush()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Handle mouse transformations
    glTranslatef(initial_translate['x'], initial_translate['y'], initial_translate['z'])
    glRotatef(rotate['x'], 1, 0, 0)
    glRotatef(rotate['y'], 0, 1, 0) #we switched around the axis so make this rotate_z
    glTranslatef(translate['x'], translate['y'], translate['z'])
    
    # Render the particles
    glEnable(GL_POINT_SMOOTH)
    glPointSize(2)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Set up the VBOs
    gl_color.bind()
    glColorPointer(4, GL_FLOAT, 0, gl_color)
    gl_position.bind()
    glVertexPointer(4, GL_FLOAT, 0, gl_position)
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)

    # Draw the VBOs
    glDrawArrays(GL_POINTS, 0, num_particles)

    glDisableClientState(GL_COLOR_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)

    glDisable(GL_BLEND)

    glutSwapBuffers()

window = glut_window()

(np_position, np_velocity, gl_position, gl_color) = initial_buffers(num_particles)

platform = cl.get_platforms()[0]
context = cl.Context(properties=[(cl.context_properties.PLATFORM, platform)] + get_gl_sharing_context_properties())  
queue = cl.CommandQueue(context)

cl_velocity = cl.Buffer(context, mf.COPY_HOST_PTR, hostbuf=np_velocity)
cl_start_position = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_position)
cl_start_velocity = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_velocity)

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

glutMainLoop()
