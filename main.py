import time
import cmath
from array import array

import numpy as np
from PIL import Image
from numba import njit
import arcade

# . . .
# . x .
# . . .

# 3 2 1
# 4 x 0
# 5 6 7


@njit
def next_pixel(a, i, j, d):
    di = [0, -1, -1, -1, 0, 1, 1, 1]
    dj = [1, 1, 0, -1, -1, -1, 0, 1]
    d = (d + 5) % 8
    for t in range(8):
        ni, nj = i + di[d], j + dj[d]
        if a[ni, nj]:
            return ni, nj, d
        d = (d + 1) % 8

@njit
def border_itarate(a, i0, j0, border):
    n, m = a.shape
    k = 0
    i, j, d = next_pixel(a, i0, j0, 0)
    while k < len(border) and not (i == i0 and j == j0):
        i, j, d = next_pixel(a, i, j, d)
        border[k, 0] = i
        border[k, 1] = j
        k += 1
    return k

class CircleAnimationHW(arcade.Window):
    def __init__(self, H, W, scale, N_STEP=4, N_FFT=96, N_LC=32):
        super().__init__(int(W*scale), int(H*scale), 'CircleAnimationHW')
        arcade.set_background_color(arcade.color.AMAZON)
        self.W, self.H = W, H
        self.scale = scale
        self.N_STEP, self.N_FFT, self.N_LC = N_STEP, N_FFT, N_LC
        self.i = None
        self.program = self.ctx.load_program(
            vertex_shader='shaders/vertex_shader.glsl',
            fragment_shader='shaders/fragment_shader.glsl',
        )
        self.buffer = None
        self.vao = None
        self.ctx.enable_only()

    def setup(self, y, y_fft):
        self.y, self.y_fft = y, y_fft
        print('y min max')
        print(self.y.view(np.float32).reshape(-1, 2).min(axis=0))
        print(self.y.view(np.float32).reshape(-1, 2).max(axis=0))
        self.i = 0
        self.buffer = self.ctx.buffer(data=array('f', self.y.view(np.float32)))
        desc = arcade.gl.BufferDescription(
            self.buffer,
            '2f',
            ['in_pos']
        )
        self.vao = self.ctx.geometry([desc])

    def on_draw(self):
        self.clear()
        self.ctx.point_size = 2 * self.get_pixel_ratio()
        self.vao.render(self.program, mode=self.ctx.POINTS)

    def on_update(self, delta_time: float):
        pass

class CircleAnimation(arcade.Window):
    def __init__(self, H, W, scale, y, y_fft, N_STEP=4, N_FFT=96, N_LC=32):
        super().__init__(int(W*scale), int(H*scale), 'CircleAnimation')
        arcade.set_background_color(arcade.color.AMAZON)
        self.W = W
        self.H = H
        self.scale = scale
        self.y = y
        self.y_fft = y_fft
        self.N_STEP = N_STEP
        self.N_FFT = N_FFT
        self.N_LC = N_LC
        self.i = 0
        self.j = 0
        self.cum_time = .0
        self.ifps = 1 / 60.
        self.frame_count = 0
        self.render_time = .0
        self.update_time = .0
        self.y_shapes = arcade.ShapeElementList()
        self.fft_y_shapes = arcade.ShapeElementList()
        self.fft_y_lc_shapes = arcade.ShapeElementList()
        self.circles_shapes = arcade.ShapeElementList()
        self.draw_start = None
        self.last_x = None
        self.last_y = None
        self.fft_last_x = None
        self.fft_last_y = None
        self.fft_lc_last_x = None
        self.fft_lc_last_y = None

    def apply_scale(self, x, y):
        return x * self.H * self.scale, y * self.H * self.scale

    def xy_from_c(self, c):
        y, x = 1.0 - c.real, c.imag
        x, y = self.apply_scale(x, y)
        return x, y

    def on_draw(self):
        draw_start = time.time()
        arcade.start_render()
        while self.i < self.j:
            c = self.y[self.i]
            assert isinstance(c, np.complex64)
            x, y = self.xy_from_c(c)
            if self.i % self.N_STEP == 0:
                last_draw = self.i + self.N_STEP >= self.j
                if self.last_x is not None:
                    line = arcade.create_line(self.last_x, self.last_y, x, y, arcade.color.AERO_BLUE, 4)
                    self.y_shapes.append(line)
                self.last_x, self.last_y = x, y
                if last_draw:
                    self.circles_shapes = arcade.ShapeElementList()
                c = self.y_fft[0]
                cwt = cmath.rect(1, cmath.pi * self.i * 2 / len(self.y_fft))
                ct = cwt
                for k in range(self.N_FFT):
                    line_color = (255, 0 + 255 * k // self.N_FFT, 0 + 255 * k // self.N_FFT)
                    circle_color = (220, 230, 255 * k // self.N_LC)
                    if last_draw:
                        cx0, cy0 = self.xy_from_c(c)
                    c += self.y_fft[k + 1] * ct
                    if last_draw:
                        cx1, cy1 = self.xy_from_c(c)
                        line = arcade.create_line(cx0, cy0, cx1, cy1, line_color)
                        self.circles_shapes.append(line)
                        if k < self.N_LC:
                            r = np.absolute(self.y_fft[k + 1]) * self.H * self.scale
                            circle = arcade.create_ellipse_outline(cx0, cy0, 2 * r, 2 * r,
                                circle_color, tilt_angle=90+cmath.phase(ct)*180/cmath.pi, num_segments=max(8, 32 - k))
                            self.circles_shapes.append(circle)
                    c += self.y_fft[-k - 1] * ct.conjugate()
                    if last_draw:
                        cx2, cy2 = self.xy_from_c(c)
                        line = arcade.create_line(cx1, cy1, cx2, cy2, line_color)
                        self.circles_shapes.append(line)
                        if k < self.N_LC:
                            r = np.absolute(self.y_fft[-k - 1]) * self.H * self.scale
                            circle = arcade.create_ellipse_outline(cx1, cy1, 2 * r, 2 * r,
                                circle_color, tilt_angle = 90-cmath.phase(ct) * 180 / cmath.pi, num_segments = max(8, 32 - k))
                            self.circles_shapes.append(circle)
                    if k + 1 == self.N_LC:
                        c_lc = c
                    ct *= cwt
                x, y = self.xy_from_c(c)

                if self.fft_last_x is not None:
                    line = arcade.create_line(self.fft_last_x, self.fft_last_y, x, y, arcade.color.ROSE, 4)
                    self.fft_y_shapes.append(line)
                self.fft_last_x, self.fft_last_y = x, y

                x, y = self.xy_from_c(c_lc)
                if self.fft_lc_last_x is not None:
                    line = arcade.create_line(self.fft_lc_last_x, self.fft_lc_last_y, x, y, arcade.color.ALLOY_ORANGE, 4)
                    self.fft_y_lc_shapes.append(line)
                    pass
                self.fft_lc_last_x, self.fft_lc_last_y = x, y
            self.i += 1
        update_time = time.time() - draw_start
        self.y_shapes.draw()
        self.fft_y_shapes.draw()
        self.fft_y_lc_shapes.draw()
        if self.j < len(self.y_fft):
            self.circles_shapes.draw()
        draw_end = time.time()

        self.frame_count += 1
        between_frames = .0
        if self.draw_start is not None:
            between_frames = draw_start - self.draw_start
            self.ifps += (between_frames - self.ifps) * .1
        self.draw_start = draw_start

        render_time = draw_end - draw_start

        self.render_time += (render_time - self.render_time) * .1
        self.update_time += (update_time - self.update_time) * .1
        text_pos = self.H * self.scale
        arcade.draw_text(f'FPS: {1 / self.ifps:.0f}', 20, text_pos := text_pos - 40, arcade.color.WHITE, 32)
        arcade.draw_text(f'render_time: {self.render_time:.3f}', 20, text_pos := text_pos - 40, arcade.color.WHITE, 32)
        arcade.draw_text(f'update_time: {self.update_time:.3f}', 20, text_pos := text_pos - 40, arcade.color.WHITE, 32)
        arcade.draw_text(f'between_frames: {between_frames:.3f}', 20, text_pos := text_pos - 40, arcade.color.WHITE, 32)
        arcade.draw_text(f'frame_count: {self.frame_count:7d}', 20, text_pos := text_pos - 40, arcade.color.WHITE, 32)
        arcade.draw_text(f'output_pos: {self.j:7d}', 20, text_pos := text_pos - 40, arcade.color.WHITE, 32)
        arcade.draw_text(f'y_shapes: {len(self.y_shapes):7d}', 20, text_pos := text_pos - 40, arcade.color.WHITE, 32)
        arcade.draw_text(f'fft_y_shapes: {len(self.fft_y_shapes):7d}', 20, text_pos := text_pos - 40, arcade.color.WHITE, 32)
        arcade.draw_text(f'circles_shapes: {len(self.circles_shapes):7d}', 20, text_pos := text_pos - 40, arcade.color.WHITE, 32)

    def on_update(self, delta_time: float):
        speed = 0.001
        self.cum_time += delta_time
        dj = int(self.cum_time // speed)
        # dj = int(min(self.cum_time // speed, self.N_STEP))
        self.cum_time -= dj * speed
        self.j = min(self.j + dj, len(self.y))

def main(filename='wold-map-4689484.jpg'):
    with Image.open(filename) as im:
        a = np.all(np.array(im) > 127, axis=-1)
        a = a[32:-32, 32:-32]
    assert isinstance(a, np.ndarray)
    # Image.fromarray(a).show()
    print(a.shape, a.dtype)
    a ^= True
    i, j = np.nonzero(a)
    assert isinstance(i, np.ndarray) and isinstance(j, np.ndarray)
    j0, j1 = j.min(), j.max()
    i0, i1 = i[j == j0].min(), i[j == j1].max()
    border: np.ndarray = np.empty(shape=(100000, 2), dtype=np.int32)
    k = border_itarate(a, i0, j0, border)
    border = border[:k]
    # border = border[:(border == [i1, j1]).all(axis=1).nonzero()[0][0]]
    assert np.all(a[border[:, 0], border[:, 1]])
    iborder = border.copy()
    print(np.c_[iborder.min(axis=0), iborder.max(axis=0)])
    y = (border.astype(np.float32) / a.shape[0]).view(dtype=np.complex64).ravel()
    iy = y.copy()
    y = np.fft.fft(y, norm='forward')

    N_FFT = 192
    y[1 + N_FFT: -N_FFT] = 0

    animation = CircleAnimation(a.shape[0], a.shape[1], .5, iy, y, N_FFT=N_FFT)
    # animation = CircleAnimationHW(a.shape[0], a.shape[1], .5)
    # animation.setup(iy, y)

    y = np.fft.ifft(y, norm='forward')
    border = (y.astype(np.complex64).view(np.float32).reshape(-1, 2) * a.shape[0]).astype(np.int32)

    print(np.c_[border.min(axis=0), border.max(axis=0)])
    border[:, 0] = np.clip(border[:, 0], 0, a.shape[0] - 1)
    border[:, 1] = np.clip(border[:, 1], 0, a.shape[1] - 1)

    print(np.sqrt(np.square(iborder - border).sum(axis=-1)).mean())
    arcade.run()
    # b = np.zeros_like(a)
    # b[border[:, 0], border[:, 1]] = 1
    # b[iborder[:, 0], iborder[:, 1]] = 1
    # Image.fromarray(b).show()

    # b: np.ndarray = np.repeat(np.expand_dims(255 - a.astype(np.uint8) * 255, axis=2), 3, axis=2)
    # b[border[:, 0], border[:, 1], 0] = 255
    # Image.fromarray(b, mode='RGB').show()


if __name__ == '__main__':
    main()