import numpy as np
from functools import partial
import vispy
from vispy import app, gloo, visuals, scene

vertex_shader = """ 
attribute vec3 position;
attribute vec4 color;
attribute float pt_size;
varying vec4 color_gg;
void main()
{
    vec4 visual_pos = vec4(position, 1);
    vec4 doc_pos = $visual_to_doc(visual_pos);
    gl_Position = $doc_to_render(doc_pos);
    color_gg = color;
    gl_PointSize = pt_size;
}
"""
fragment_shader = """ 
varying vec4 color_gg;
void main() {
  gl_FragColor = color_gg;
}
"""


class Plot3DVisual(visuals.Visual):
    def __init__(self, xyz, rgb=None, pt_size=1):
        visuals.Visual.__init__(self, vertex_shader, fragment_shader)

        v = gloo.VertexBuffer(xyz.astype(np.float32))
        c = gloo.VertexBuffer(rgb.astype(np.float32))
        self.shared_program['position'] = v
        self.shared_program['color'] = c
        self.shared_program['pt_size'] = pt_size

        self.set_gl_state('opaque', depth_test=True)
        self._draw_mode = 'points'

    def _prepare_transforms(self, view):
        tr = view.transforms
        view_vert = view.view_program.vert
        view_vert['visual_to_doc'] = tr.get_transform('visual', 'document')
        view_vert['doc_to_render'] = tr.get_transform('document', 'render')

    def on_draw(self, event):
        gloo.clear()
        self.program.draw('points')


def CreateView(xyz, rgb, pt_size=1, fov=60, distance=30):
    Plot3D = scene.visuals.create_visual_node(Plot3DVisual)
    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor=(0, 0, 0, 1))
    view = canvas.central_widget.add_view()
    view.camera = 'arcball'
    view.camera.fov = fov
    view.camera.distance = distance
    
    [h, c] = rgb.shape
    assert c == 3
    ones = np.ones([h, 1])
    rgb = np.concatenate([rgb, ones], axis=1)
    p1 = Plot3D(xyz, rgb, pt_size=pt_size, parent=view.scene)
    return p1


def SphereGrid(equ_h, equ_w):
    cen_x = (equ_w - 1) / 2.0
    cen_y = (equ_h - 1) / 2.0
    theta = (2 * (np.arange(equ_w) - cen_x) / equ_w) * np.pi
    phi = (2 * (np.arange(equ_h) - cen_y) / equ_h) * (np.pi / 2)
    theta = np.tile(theta[None, :], [equ_h, 1])
    phi = np.tile(phi[None, :], [equ_w, 1]).T

    x = (np.cos(phi) * np.sin(theta)).reshape([equ_h, equ_w, 1])
    y = (np.sin(phi)).reshape([equ_h, equ_w, 1])
    z = (np.cos(phi) * np.cos(theta)).reshape([equ_h, equ_w, 1])
    xyz = np.concatenate([x, y, z], axis=-1)

    return xyz

