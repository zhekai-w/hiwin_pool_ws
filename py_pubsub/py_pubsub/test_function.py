import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon

# Create a polygon
polygon_coords = [(0, 0), (0, 10), (10, 10), (10, 0)]
polygon = Polygon(polygon_coords)

# Create a line
line_coords = [(15, 1), (15, 8)]
line = LineString(line_coords)

# Check if the line is inside the polygon
is_line_inside_polygon = line.intersects(polygon)

# Plot the polygon and the line
x_poly, y_poly = zip(*polygon.exterior.coords)
x_line, y_line = zip(*line.coords)

plt.plot(x_poly, y_poly, label='Polygon')
plt.plot(x_line, y_line, label='Line')

# Highlight the line if it's inside the polygon
if is_line_inside_polygon:
    plt.fill(x_line, y_line, color='gray', alpha=0.5, label='Inside Polygon')
    print("sdasdasd")

# Customize the plot
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Inside Polygon Visualization')
plt.legend()
plt.grid(True)
plt.show()
