import matplotlib.colors as cl
import colorsys as cs

def get_colormap(num_colors, start=0.0, satu=1.0, val=1.0):
	""" 
	Get a color map that has colors equally spaced on color circle.

	Args:
		num_colors: Number of colors required
		start: Hue of the first color
		satu: Saturation of all colors
		val: Value of all colors
	Return:
		A color map in type of matplotlib.colors.ListedColormap
	"""
	color_list = []
	for i in range(0, num_colors):
		col = cs.hsv_to_rgb(float(i)/num_colors+start, satu, val)
		color_list.append('#' + ''.join([format(int(c*255), '02x') for c in col]))
		#print(col, color_list)
	return cl.ListedColormap(color_list)
