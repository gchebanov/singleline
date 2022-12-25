#version 330

in vec2 in_pos;
out vec4 color;

void main() {
    color = vec4(1, 1, 1, 1);
    float ratio = 16 / 9.;
    vec2 pos = in_pos.yx;
    pos.x /= ratio;
    pos = pos * 2 - 1;
    pos.y = -pos.y;
    gl_Position = vec4(pos, 0, 1);
}
