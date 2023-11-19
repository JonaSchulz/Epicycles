import numpy as np
import pygame
import math
import cmath
import fractions

black = 0, 0, 0
white = 255, 255, 255
white_2 = 100, 100, 100
blue = 40, 100, 100
red = 180, 40, 40
grey = 60, 60, 60
size = 1200, 900
background_color = black

number_of_arrows = 50
unit = 100              # number of pixels corresponding to 1 on the complex plane
arrow_color = red
path_color = blue
circle_color = grey
path_width = 3
sample_spacing = 2      # max distance between two neighbouring sample points (in pixels)

# function for width of an arrow based on its length:
def arrow_width(length):
    if length > 1:
        return 4
    elif length > 0.7:
        return 3
    elif length > 0.1:
        return 2
    else:
        return 1

class Fourier:
    """ class containing functions related to the Fourier transform """

    @classmethod
    def dft(cls, array):
        """ computes the discrete Fourier transform for an array of numbers, returns list of tuples of type (k, a_k)
            with k = multiple of base frequency 1/N, a_k = complex coefficient of arrow rotating with frequency k/N: """
        return [(k, np.fft.fft(array)[k]) for k in range(len(array))]

        length = len(array)
        result = np.zeros(length).astype(complex)
        for k in range(length):
            for n in range(length):
                result[k] += array[n] * (cmath.exp(-2j * cmath.pi / length * n * k))
        return [(k, result[k]) for k in range(len(result))]

    @classmethod
    def get_approx(cls, dft, n):
        """ returns a list of (k, a_k) tuples for the n arrows with greatest magnitude """
        sorted_dft = sorted(dft, key=lambda x: -abs(x[1]))
        return sorted_dft[:n]

    @classmethod
    def create_arrows(cls, dft, fourier_sum, n):
        """ converts (k, a_k) tuples of dft into Arrow objects and adds them to a FourierSum object
            (number of sampled points n)"""
        for k in range(len(dft)):
            a_k = dft[k][1] / n
            length = abs(a_k)
            phi = cmath.phase(a_k)
            fourier_sum.add_arrow(length, fractions.Fraction(dft[k][0]) / n, phi)


class Arrow:
    """ class for a single arrow, stores frequency, angle and position of arrow on the x-y plane;
        arrows can rotate and be drawn onto a screen"""

    def __init__(self, length, freq, phi_0, origin):
        self.length = length
        self.freq = freq    # frequency of rotation
        self.phi = phi_0    # current angle
        self.origin = origin    # current location of arrow origin on x-y plane
        self.end = self.origin + self.length * np.array([math.cos(self.phi), math.sin(self.phi)])   # current location of arrow tip on x-y plane
        self.tip = self.get_triangle(self.get_tip_length())     # list of positions of triangle making up the arrow's tip

    def rotate(self, dt):
        """ rotates the arrow for a time of dt based on its frequency """
        self.phi += 2 * math.pi * float(self.freq) * dt
        self.end = self.origin + self.length * np.array([math.cos(self.phi), math.sin(self.phi)])   # update arrow end
        self.tip = self.get_triangle(self.get_tip_length())    # update arrow tip triangle

    def draw(self, screen, color, unit, screen_origin):
        """ draws the arrow onto the screen based on number of pixels per unit """
        x = self.length * math.cos(self.phi)
        y = -self.length * math.sin(self.phi)   # negative y coordinate because of opposite orientation of y axis in pygame
        origin = np.array([self.origin[0], -self.origin[1]])
        end_pos = screen_origin + unit * (origin + np.array([x, y]))    # arrow tip position in pixel coordinates
        start_pos = screen_origin + unit * origin       # arrow origin position in pixel coordinates
        pygame.draw.line(screen, color, tuple(start_pos), tuple(end_pos), arrow_width(self.length))

        # convert tip triangle positions to pixel coordinates and draw arrow tip:
        tip = []
        for point in range(len(self.tip)):
            tip.append(np.array([int(unit * self.tip[point][0]), int(-unit * self.tip[point][1])]) + np.asarray(screen_origin))
        pygame.draw.polygon(screen, color, tip)

    def draw_circle(self, screen, color, unit, screen_origin):
        """ draws a circle with radius = length of arrow centered around the arrow origin onto the screen """
        radius = int(unit * self.length)
        if radius <= 1:
            return
        pos = screen_origin + unit * np.array([self.origin[0], -self.origin[1]])
        pos = (int(pos[0]), int(pos[1]))
        pygame.draw.circle(screen, color, pos, radius, 1)

    def set_end(self):
        """ sets end position based on origin, angle and length (called during initialization of FourierSum """
        self.end = self.origin + self.length * np.array([math.cos(self.phi), math.sin(self.phi)])

    def get_triangle(self, tip_len):
        """ returns coordinates in the x-y plane for arrow tip triangle """
        vector = self.end - self.origin
        d = tip_len
        h = math.sqrt(d ** 2 - (d / 2) ** 2)
        v_unit = vector / self.length
        v_unit_normal = np.array([-v_unit[1], v_unit[0]])
        b = self.end - h * v_unit
        c = b + d / 2 * v_unit_normal
        d = b - d / 2 * v_unit_normal
        return [self.end, c, d]

    def get_tip_length(self):
        """ function to compute length of triangle sides for arrow tip """
        return self.length / 10


class FourierSum:
    """ class for storing and coordinating a sum of arrows; rotates all arrows, updates their positions and draws them"""

    def __init__(self):
        self.arrows = []
        self.end = np.array([0, 0])
        self.path = []
        self.freq_0 = 0
        self.clock = 0

    def add_arrow(self, length, freq, phi_0):
        """ adds an arrow to the sum; resorts all arrows and adjusts their positions accordingly """
        self.arrows.append(Arrow(length, freq, phi_0, np.array([0, 0])))
        self.arrows.sort(key=lambda x: -x.length)  # sorting based on arrow length from greatest to smallest

        for i in range(len(self.arrows)):
            origin = FourierSum.sum_arrows(self.arrows[:i])     # compute new origin of each arrow
            self.arrows[i].origin = origin  # adjust arrow origin
            self.arrows[i].set_end()    #adjust arrow end

        self.end = self.arrows[-1].end  # tip position of last arrow
        self.freq_0 = self.get_freq_0()     # base frequency (gcd of all occurring frequencies)

    def update(self, dt):
        """ update all arrows by time step dt """
        if len(self.arrows) == 0:
            return
        self.clock += dt
        for i in range(len(self.arrows)):
            self.arrows[i].rotate(dt)
            if i + 1 < len(self.arrows):
                self.arrows[i + 1].origin = self.arrows[i].end  # update arrow positions
        self.end = self.arrows[-1].end  # update total tip position
        if self.clock <= 1 / self.freq_0:
            self.path.append(self.end)  # only store path coordinates in first period

    def draw_arrows(self, screen, color, unit, screen_origin):
        """ draws all arrows onto the screen """
        for arrow in self.arrows:
            arrow.draw(screen, color, unit, screen_origin)

    def draw_circles(self, screen, color, unit, screen_origin):
        """ draws circles around all arrows """
        for arrow in self.arrows:
            arrow.draw_circle(screen, color, unit, screen_origin)

    def draw_path(self, screen, color, unit, screen_origin, width):
        """ draws path traced by tip of arrow sum """
        for i in range(len(self.path) - 1):
            x1_pos = screen_origin[0] + self.path[i][0] * unit
            y1_pos = screen_origin[1] - self.path[i][1] * unit
            x2_pos = screen_origin[0] + self.path[i + 1][0] * unit
            y2_pos = screen_origin[1] - self.path[i + 1][1] * unit
            pygame.draw.line(screen, color, (x1_pos, y1_pos), (x2_pos, y2_pos), width)

    def get_freq_0(self):
        """ computes base frequency as gcd of all occurring frequencies"""
        gcd = self.arrows[0].freq
        if len(self.arrows) == 1:
            return gcd
        else:
            for i in range(len(self.arrows) - 1):
                gcd = fractions.Fraction(np.gcd(gcd.numerator, self.arrows[i + 1].freq.numerator),
                      np.lcm(gcd.denominator, self.arrows[i + 1].freq.denominator))
            return gcd

    @classmethod
    def sum_arrows(cls, arrow_lst):
        """ computes sum of all arrows in arrow_lst and returns tip position of last arrow """
        result = np.array([float(0), float(0)])
        for arrow in arrow_lst:
            result += arrow.length * np.array([math.cos(arrow.phi), math.sin(arrow.phi)])
        return result

def dist(p1, p2):
    """ returns distance between two points in the x-y plane """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_points_on_line(p1, p2, spacing):
    """ returns evenly spaced points on a line segment between points p1 and p2 """
    points = []
    if spacing == 0:
        return points
    number_of_points = int(dist(p1, p2) / spacing)
    for i in range(number_of_points):
        points.append(p1 + i / number_of_points * (p2 - p1))
        points[-1] = np.array([int(points[-1][0]), int(points[-1][1])])
    return points

def sample(screen, color, unit, screen_origin, min_dist=math.inf):
    """ allows user to input sample points by drawing on the screen"""
    end = False
    points = [[], []]
    while not end:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                if len(points[1]) > 0 and dist(pos, points[1][-1]) > min_dist:
                    for p in get_points_on_line(np.asarray(points[1][-1]), np.asarray(pos), min_dist):
                        points[1].append(tuple(p))
                        points[0].append((p[0] - screen_origin[0] - (p[1] - screen_origin[1]) * 1j) / unit)
                points[1].append(pos)
                pygame.draw.circle(screen, color, pos, 2)
                points[0].append((pos[0] - screen_origin[0] - (pos[1] - screen_origin[1]) * 1j) / unit)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    end = True
        pygame.display.flip()
    if len(points[0]) > 1 and dist(points[1][0], points[1][-1]) > min_dist:
        for p in get_points_on_line(np.asarray(points[1][-1]), np.asarray(points[1][0]), min_dist):
            points[1].append(tuple(p))
            points[0].append((p[0] - screen_origin[0] - (p[1] - screen_origin[1]) * 1j) / unit)
    return points

def main():
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Epicycles")
    clock = pygame.time.Clock()
    screen_origin = (size[0]/2, size[1]/2)
    draw = True
    pause = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    pause = not pause
                if event.key == pygame.K_c:
                    draw = True
        if draw:
            screen.fill(background_color)
            f = FourierSum()
            samples = sample(screen, white, 100, screen_origin, sample_spacing)
            dft = Fourier.dft(samples[0])
            dft_approx = Fourier.get_approx(dft, number_of_arrows)
            Fourier.create_arrows(dft_approx, f, len(dft))
            draw = False
            pause = False

        else:
            if not pause:
                screen.fill(background_color)
                f.update(1)
                f.draw_path(screen, path_color, unit, np.asarray(screen_origin), path_width)
                f.draw_circles(screen, circle_color, unit, np.asarray(screen_origin))
                f.draw_arrows(screen, arrow_color, unit, np.asarray(screen_origin))
                pygame.display.flip()
                clock.tick()
                pygame.time.wait(10 * sample_spacing - clock.get_time())
            else:
                pass

if __name__ == "__main__":
    main()
