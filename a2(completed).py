import matplotlib.pyplot as plt
import numpy as np
import random
import math

class Vec2D:
    def __init__(self, x, y):
        """Store x and y in a form of your choosing."""
        self._x = x
        self._y = y

    def __str__(self):
        """Returns string of form '(x, y)'"""
        return f'({self._x}, {self._y})'

    def __repr__(self):
        """Returns string of form 'Vec2D(x, y)'"""
        return f'Vec2D({self._x}, {self._y})'

    def get_x(self):
        """Getter for x. Returns x."""
        return self._x

    def get_y(self):
        """Getter for y. Returns y."""
        return self._y

    def set_x(self, x):
        """Setter for x."""
        self._x = x

    def set_y(self, y):
        """Setter for y."""
        self._y = y

    def __add__(self, other):
        """Vector addition of self and other. Returns a Vec2D object."""
        return Vec2D(self._x + other.get_x(), self._y + other.get_y())

    def __sub__(self, other):
        """Vector subtraction: self - other. Returns a Vec2D object."""
        return Vec2D(self._x - other.get_x(), self._y - other.get_y())

    def __mul__(self, scalar):
        """Multiplication on left of Vec2D and scalar: self * scalar.
        Returns Vec2D object."""
        return Vec2D(self._x * scalar, self._y * scalar)

    def __rmul__(self, scalar):
        """Multiplication on right of Vec2D and scalar: scalar * self.
        Returns Vec2D object."""
        return self.__mul__(scalar)

    def length(self):
        """Computes length of the vector. Returns a float."""
        return (self._x**2 + self._y**2)**0.5

    def dot(self, other):
        """Computes dot product of self and other. Returns a float."""
        return self._x * other.get_x() + self._y * other.get_y()

def closest_point_on_line_segment(p, a, b):
    """
    Calculate the closest point on a line segment to a given point.

    Parameters:
    p (numpy array): The point from which we want to find the
                                closest point on the line segment.
    a (numpy array): The first endpoint of the line segment.
    b (numpy array): The second endpoint of the line segment.

    Returns:
    numpy array: The closest point on the line segment to
                            the point `p`.
    """
    # Vector from point a to point b
    u = b - a
    # Vector from point a to point p
    ap = p - a
    # Dot product of u with itself (squared length of u)
    u_dot_u = u.dot(u)
    
    if u_dot_u == 0:
        return a

    # Projection factor of ap on u
    t = ap.dot(u) / u_dot_u
    t = max(0.0, min(1.0, t))

    # Interpolate between a and b
    closest = (1 - t) * a + t * b
    
    return closest

class Boundary:
    def __init__(self, filename):
        """
        Initialize a Boundary instance by reading the polygon vertices
        and associated temperatures from a file. The file should contain
        one vertex per line, with coordinates and temperature separated
        by spaces.

        Parameters:
        filename (str): Path to the file containing the polygon data.
        """
        self.vertices = []
        self.temperatures = []
        
        with open(filename, 'r') as file:
            for line in file:
               
                parts = line.split()
                if len(parts) == 3:
                    # Extract x, y, and temperature
                    x, y, temp = parts
                    # Create and add Vec2D for vertex
                    self.vertices.append(Vec2D(float(x), float(y)))
                    self.temperatures.append(float(temp))

    def distance_to_closest_edge(self, p):
        """
        Find the distance to the closest edge from a given point `p`.

        Parameters:
        p (Vec2D): The point from which the closest edge's distance needs
                   to be calculated.

        Returns:
        tuple: A tuple containing the index of the closest edge and the
               distance to that edge.
        """
        # Initialize with infinity
        min_distance = float('inf')
        closest_edge_index = -1

        for i in range(len(self.vertices) - 1):
            # Start vertex of the edge
            a = self.vertices[i]
            # End vertex of the edge
            b = self.vertices[i + 1]
            closest_point = closest_point_on_line_segment(p, a, b)
            distance = ((p.get_x() - closest_point.get_x())**2 +
                        (p.get_y() - closest_point.get_y())**2)**0.5

            if distance < min_distance:
                min_distance = distance
                # Update closest edge
                closest_edge_index = i

        return (closest_edge_index, min_distance)

    def temperature_at_edge(self, p, edge_index):
        """
        Calculate the interpolated temperature based on the point
        `p` and an edge index.

        Parameters:
        p (Vec2D): The point on the edge for which the temperature
                   is to be calculated.
        edge_index (int): The index of the starting vertex of the edge.

        Returns:
        float: The interpolated temperature at the point `p` on the
               specified edge.
        """
        # Start vertex of the edge
        a = self.vertices[edge_index]
        # End vertex of the edge
        b = self.vertices[edge_index + 1]
        # Temperature at vertex a
        Ta = self.temperatures[edge_index]
        # Temperature at vertex b
        Tb = self.temperatures[edge_index + 1]

        u = b - a
        ap = p - a
        u_dot_u = u.dot(u)
        ap_dot_u = ap.dot(u)

        if u_dot_u != 0:
            t = ap_dot_u / u_dot_u
        else:
            t = 0
    
        t = max(0.0, min(1.0, t))
    
        T_edge = (1 - t) * Ta + t * Tb

        return T_edge

    def wos_temperature_estimate(self, p0, epsilon, number_walks):
        """
        Estimate the average temperature using the walk-on-spheres (WOS) method.

        Parameters:
        p0 (Vec2D): Starting point for the walk-on-spheres algorithm.
        epsilon (float): The threshold distance to consider a point as being
                        near an edge.
        number_walks (int): Number of random walks to perform.

        Returns:
        tuple: Average temperature estimated using the WOS method
               and an array of temperatures found.
        """
        edge_temps = []

        for _ in range(number_walks):
            # Start at p0
            current_point = Vec2D(p0.get_x(), p0.get_y())  

            while True:
                closest_edge_index, closest_distance = \
                    self.distance_to_closest_edge(current_point)

                if closest_distance <= epsilon:
                    # Stop if within epsilon distance of an edge
                    break

                # Random angle
                angle = random.uniform(0, 2 * math.pi)
                new_x = current_point.get_x() -closest_distance *math.cos(angle)
                new_y = current_point.get_y() -closest_distance *math.sin(angle)
                # Move to new point
                current_point = Vec2D(new_x, new_y)

            edge_temp = \
                self.temperature_at_edge(current_point, closest_edge_index)
            # Record temperature
            edge_temps.append(edge_temp)

        if edge_temps:
            # Average temperature
            T_avg = sum(edge_temps) / len(edge_temps)
        else:
            T_avg = 0.0

        return T_avg, np.array(edge_temps)

def plot_temperature_convergence(Ts):
    """
    Plots the temperatures from a series of walks and shows
    the running average to visualize the convergence pattern
    during a walk-on-spheres simulation.

    Parameters:
    Ts (list or numpy array): List or array of temperature values from
                              each walk.
    """
    cumulative_sum = 0
    cumulative_averages = []

    for i, temp in enumerate(Ts):
        cumulative_sum += temp
        # Running average
        cumulative_averages.append(cumulative_sum / (i + 1))

    walk_numbers = np.arange(1, len(Ts) + 1)

    plt.plot(walk_numbers, Ts, 'o', color='#0000ff',
             label='Edge Temperature per Walk', markersize=3, linestyle='None',
             linewidth=1.5)
    plt.plot(walk_numbers, cumulative_averages, label='Running Average',
             linestyle='-', color='red')

    plt.title('Temperature Convergence during Walk-On-Spheres Simulation')
    plt.xlabel('Walk Number')
    plt.ylabel('Temperature (K)')

    plt.legend(loc='upper right', frameon=True)

    plt.show()
    
def nearest_edge_square(x, y, length):
    """
    Finds the nearest edge of a square from a given point inside the square.

    Parameters:
        x (float): The x-coordinate of the point inside the square.
        y (float): The y-coordinate of the point inside the square.
        length (float): The length of the square's sides.

    Returns:
        tuple: A tuple containing the index of the nearest edge (0 for bottom,
        1 for right, 2 for top, 3 for left) and the distance from the point
        to the nearest edge.
    """
    dist_left = x
    dist_right = length - x
    dist_bottom = y
    dist_top = length - y

    distances = [dist_bottom, dist_right, dist_top, dist_left]
    min_dist = min(distances)

    # Index of nearest edge
    nearest_edge = distances.index(min_dist)

    return (nearest_edge, min_dist)

def temperature_at_nearest_edge(x, y, length, edge, corner_temps):
    """
    Calculates the temperature at the nearest edge of a square from a given
    point inside the square, based on the temperatures at the square's corners.

    Parameters:
        x (float): The x-coordinate of the point inside the square.
        y (float): The y-coordinate of the point inside the square.
        length (float): The length of the square's sides.
        edge (int): The index of the nearest edge (0 for south, 1 for east,
        2 for north, 3 for west).
        corner_temps (dict): A dictionary containing temperatures at the
        square's corners, with keys 'sw' for southwest, 'se' for southeast,
        'ne' for northeast, and 'nw' for northwest.

    Returns:
        float: The temperature at the nearest edge of the square from
        the given point.
    """
    # South edge (bottom)
    if edge == 0:  
        fraction = x / length
        temp = (1 - fraction) *corner_temps['sw'] +fraction *corner_temps['se']

    # East edge (right)
    elif edge == 1:  
        fraction = y / length
        temp = (1 - fraction) *corner_temps['se'] +fraction *corner_temps['ne']

    # North edge (top)
    elif edge == 2:  
        fraction = (length - x) / length
        temp = (1 - fraction) *corner_temps['ne'] +fraction *corner_temps['nw']

    # West edge (left)
    elif edge == 3:  
        fraction = (length - y) / length
        temp = (1 - fraction) *corner_temps['sw'] +fraction *corner_temps['nw']

    return temp

def jump_on_circle(x, y, radius):
    """
    Jumps randomly on a circle with the given center coordinates and radius.

    Parameters:
        x (float): The x-coordinate of the center of the circle.
        y (float): The y-coordinate of the center of the circle.
        radius (float): The radius of the circle.

    Returns:
        tuple: A tuple containing the new coordinates after the jump.
    """
    
    angle = random.uniform(0, 2 * math.pi)
    x_circ = x - radius * math.cos(angle)
    y_circ = y - radius * math.sin(angle)

    return x_circ, y_circ

def wos_temperature_estimate(x0, y0, length, cnr_temps, epsilon, number_walks):
    """
    Estimates the average temperature experienced by a point inside
    a square using the Walk-On-Spheres (WOS) method.

    Parameters:
        x0 (float): The initial x-coordinate of the point inside the square.
        y0 (float): The initial y-coordinate of the point inside the square.
        length (float): The length of the square's sides.
        cnr_temps (dict): A dictionary containing temperatures at the
                          square's corners, with keys 'sw' for southwest,
                          'se' for southeast, 'ne' for northeast, and
                          'nw' for northwest.
        epsilon (float): The minimum distance threshold from the edges
                         to stop the simulation.
        number_walks (int): The number of WOS simulations to perform.

    Returns:
        tuple: A tuple containing the average temperature estimate and
               a list of edge temperatures encountered during
               the simulations.
    """
    edge_temps = []
    
    for _ in range(number_walks):
        # Start at initial point
        x, y = x0, y0  

        while True:  

            if (x <= epsilon or 
                y <= epsilon or 
                length - x <= epsilon or 
                length - y <= epsilon):
                break  

            # Random jump
            dx, dy = random.choice([-1, 0, 1]), random.choice([-1, 0, 1])  

            x += dx  
            y += dy  

        nearest_edge, _ = nearest_edge_square(x, y, length)

        edge_temp = temperature_at_nearest_edge(
            x, y, length, nearest_edge, cnr_temps)

        edge_temps.append(edge_temp)
        
    if edge_temps:  
        avg_temp_estimate = sum(edge_temps) / len(edge_temps)  
    else:
        avg_temp_estimate = 0
        
    return avg_temp_estimate, edge_temps

def plot_temperature_histogram(Ts, bins, range):
    """
    Plots a histogram of temperatures.

    Parameters:
        Ts (list): A list of temperatures to plot.
        bins (int or sequence of scalars or str): Specification of
        histogram bins.
        range (tuple or None): The lower and upper range of the bins.

    Returns:
        None
    """
    plt.figure()
    # Plot histogram
    plt.hist(Ts, bins=bins, range=range, edgecolor='black') 
    plt.xlabel('Temperature, K')
    plt.ylabel('Count')
    plt.title('Temperature Histogram')
    plt.show()

# Example
hyp_bndry = Boundary("hyp-veh-cross-sect.dat")
p0 = Vec2D(0.23, 0.005)
epsilon = 1.0e-5
number_walks = 500
random.seed(1)
(T_avg, Ts) = hyp_bndry.wos_temperature_estimate(p0, epsilon, number_walks)
T_avg
plot_temperature_convergence(Ts)

