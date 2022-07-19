import pandas as pd
import folium
import matplotlib.pyplot as plt
import math
import re

epsilon = 1/1000 #in km
trace_file_name = "./mydata.txt"
do_rdp = False
count = 0
pd.options.mode.chained_assignment = None  

#convert degree minute second to decimal notation
def dms2dec(a):
	lat = float(a)
	d = lat // 100
	m = lat % 100
	a = float(d) + float(m) / 60
	
	return a
	
# get position and speed data
def get_pos(data):
	if len(data) < 11:
		return
	
	for i in [2,3,4,5,6,9,11]:
		if data[i] == "":
			return
	
	
	# calculate latitude
	if data[3] == 'N':
		latitude = dms2dec(data[2])
	elif data[3] == 'S':
		latitude = -dms2dec(data[2])
	
	# calculate longitude
	if data[5] == 'E':
		longitude = dms2dec(data[4])
	elif data[5] == 'W':
		longitude = -dms2dec(data[4])
		
	# calculate altitude
	altitude = float(data[9]) + float(data[11])
	
	# calculate time in sec
	h = float(data[1]) // 10000
	m = (float(data[1]) // 100) % 100 
	s = float(data[1]) % 100
	run_time = h*3600 + m*60 + s
	
	#calulate time string
	actual_time = str(int(h)) + " Hrs " + str(int(m)) + " min " + str(int(s)) + " sec"
	
	# initialize totatl distance and speed
	TDistance = 0.0
	speed = 0.0
	
	return [latitude, longitude, altitude, run_time, TDistance, speed, actual_time]

def checksum(line) :
	# checksum value
	cksum = line[line.find("*")+1:]
	
	#checksum to be calculated
	chksumdata = re.sub("(\n|\r\n)","", line[line.find("$")+1:line.find("*")])
	
	# Find checksum using or operation
	csum = 0 
	for c in chksumdata:
		csum ^= ord(c)
	
	# validate calculated checksum with given checksum
	if hex(csum) == hex(int(cksum, 16)):
		return True
	else:
		return False

def deg2rad(deg):
	return deg * (math.pi/180)
	
# get spherical Distance using haversine algorithm		 
def getDistance(lat1,lon1,alt1,lat2,lon2,alt2):
	R = 6371 # Radius of the earth in km
	dLat = deg2rad(lat2-lat1)
	dLon = deg2rad(lon2-lon1) 
	a = math.sin(dLat/2) * math.sin(dLat/2) +math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)) 
	d = R * c # Distance in km
	h = abs(alt1 - alt2) / 1000
	d = math.sqrt(d**2 + h**2)
	return d

# Convert nmea files into Dataframe
def read_data(file_name):
	f = open(file_name, 'r')
	gps_data = []
	
	for line in f.readlines():
		try:
			data = line.split(",")
			if data[0] == "$GPGGA" and checksum(line):
				temp = get_pos(data)
				
				if temp != None:
					gps_data.append(temp)
		except Exception as e:
			print('line\n' + 'error: {}'.format(e))
			continue   
	
	columns = ['Latitude', 'Longitude', 'Altitude', 'Time', 'TDistance', 'Speed', 'Actual_Time']			
	df = pd.DataFrame(gps_data, columns=columns)
	f.close()
	
	# calculate Total Distance
	for i in range(1, len(df)):
		temp_TDistance = getDistance(df['Latitude'][i-1], df['Longitude'][i-1], df['Altitude'][i-1], df['Latitude'][i], df['Longitude'][i-1], df['Altitude'][i-1])
		temp_speed = temp_TDistance * 1000 / (df['Time'][i] - df['Time'][i-1])
		
		df['TDistance'][i] = df['TDistance'][i-1] + temp_TDistance
		df['Speed'][i] = temp_speed
	
	global count
	count = len(df)
	return df

def point_line_distance(point, start, end):
	if (start == end):
		return math.sqrt((point[0] - start[0]) ** 2 + (point[1] - start[1]) ** 2)
	else:
		#abs(Cross Product(p2-p1, p1-p3)) / norm(p2-p1))
		n = abs((end[0] - start[0]) * (start[1] - point[1]) - (start[0] - point[0]) * (end[1] - start[1]))
		d = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
		return n / d


def rdp(df, epsilon):
	dmax = 0.0
	index = 0
	for i in range(1, len(df) - 1):
		d = point_line_distance([df['Latitude'].iloc[i], df['Longitude'].iloc[i]], [df['Latitude'].iloc[0], df['Longitude'].iloc[0]], [df['Latitude'].iloc[-1], df['Longitude'].iloc[-1]])
		if d > dmax:
			index = i
			dmax = d
	
	if dmax >= epsilon:
		result1 = rdp(df[:index+1], epsilon)[:-1]
		result2 = rdp(df[index:], epsilon)
		results = pd.concat([result1, result2])
	else:
		results = df.iloc[[0, len(df) - 1]]
			
	return results

def plot_data(df):
	#plot speed
	plt.plot(df['Time'] - df['Time'][0], df['Speed'])
	plt.title("Average Speed:" + str(round(sum(df['Speed']) / len(df), 2)) + " m/s")
	plt.ylabel("Speed(m/s)")
	plt.xlabel("Time (s)")
	plt.draw()
	plt.savefig('templates/speed.png')

	#clear plot
	plt.clf()

	#plot Total Distance	
	plt.plot(df['Time'] - df['Time'][0], df['TDistance'])
	plt.title("Total Distance:" + str(round(df['TDistance'].iloc[-1],2)) + "km")
	plt.xlabel("Time")
	plt.ylabel("Total Distance(km)")
	plt.draw()
	plt.savefig('templates/Total Distance.png')

def create_map(df):
	# Initiate the Folium map
	mymap = folium.Map( location=[ df.Latitude.mean(), df.Longitude.mean() ], zoom_start=16, tiles=None)
	folium.TileLayer('openstreetmap', name='OpenStreet Map').add_to(mymap)
	folium.TileLayer('Stamen Toner', name='Stamen Toner').add_to(mymap)
	folium.TileLayer('Stamen Terrain', name='Stamen Terrain').add_to(mymap)

	# Draw route using points
	folium.PolyLine(list(zip(df.Latitude, df.Longitude)), color='red', weight=4.5, opacity=.5).add_to(mymap)

	# green start circle
	iframe = folium.IFrame(f"<h3>Starting Point</h3> <a href=plot.html target=_blank> Click To See Plots </a> <b>Latitude:</b> <br /> {round(df['Latitude'][0], 5)} </br> <b>Longitude :</b> {round(df['Longitude'][0], 5)} </br> <b>Altitude :</b> {round(df['Altitude'][0])} </br> <b>Speed :</b> {round(df['Speed'][0], 2)} m/s <br/> <b>Distance Travelled :</b> {round(df['TDistance'][0], 2)} km <br /> <b>Clock Time :</b> {df['Actual_Time'][0]} ")
	popup = folium.Popup(iframe, min_width=200, max_width=400, min_height=400, max_height=600)
	folium.vector_layers.CircleMarker(location=[df['Latitude'][0], df['Longitude'][0]], radius=9, color='white', weight=1, fill_color='green', fill_opacity=1, popup=popup).add_to(mymap) 
	folium.RegularPolygonMarker(location=[df['Latitude'][0], df['Longitude'][0]], fill_color='white', fill_opacity=1, color='white', number_of_sides=3, radius=3, rotation=0).add_to(mymap)

	# red stop circle
	iframe = folium.IFrame(f"<h3>Stoping Point</h3> <a href=plot.html target=_blank> Click To See Plots </a> <br /> <b>Latitude:</b> {round(df['Latitude'].iloc[-1], 5)} </br> <b>Longitude :</b> {round(df['Longitude'].iloc[-1], 5)} </br> <b>Altitude :</b> {round(df['Altitude'].iloc[-1], 3)} </br> <b>Speed :</b> {round(df['Speed'].iloc[-1], 2)} m/s <br/> <b>Distance Travelled :</b> {round(df['TDistance'].iloc[-1], 2)} km <br /> <b>Clock Time :</b> {df['Actual_Time'].iloc[-1]}")
	popup = folium.Popup(iframe, min_width=200, max_width=400, min_height=400, max_height=600)
	folium.vector_layers.CircleMarker(location=[df['Latitude'].iloc[-1], df['Longitude'].iloc[-1]], radius=9, color='white', weight=1, fill_color='red', fill_opacity=1, popup=popup).add_to(mymap) 
	folium.RegularPolygonMarker(location=[df['Latitude'].iloc[-1], df['Longitude'].iloc[-1]], fill_color='white', fill_opacity=1, color='white', number_of_sides=4, radius=3, rotation=45).add_to(mymap)

	# ploting data points circle
	for i in range(1, len(df) - 1):
		iframe = folium.IFrame(f"<h3>GPS Datapoint</h3> <b>Latitude:</b> {round(df['Latitude'][i], 5)} </br> <b>Longitude :</b> {round(df['Longitude'][i], 5)} </br> <b>Altitude :</b> {round(df['Altitude'][i], 3)} </br> <b>Speed :</b> {round(df['Speed'][i], 2)} m/s <br/> <b>Distance Travelled :</b> {round(df['TDistance'][i], 2)} km  <br /> <b>Clock Time :</b> {df['Actual_Time'][0]}")
		popup = folium.Popup(iframe, min_width=200, max_width=400, min_height=400, max_height=600)
		folium.CircleMarker(location=[df['Latitude'][i], df['Longitude'][i]], radius=4, popup=popup, color="#0000FF", fill=True, fill_color="#000000",).add_to(mymap)
		
	# adding control
	folium.LayerControl(collapsed=True).add_to(mymap)
	
	# To enable lat/lon popovers
	mymap.add_child(folium.LatLngPopup())

	# Output map file
	mymap.save("templates/mymap.html")


# init part 
def main():
	#File to analysis
	#trace_file_name = input('Enter nmea file name: ')

	# loading file	
	#df = read_data("./" + trace_file_name)
	df = read_data(trace_file_name)
	
	# check for curve smoothing
	if do_rdp:
		df = rdp(df, epsilon)
		df.reset_index(inplace = True)

	# create html map
	create_map(df)
	
	# plotting Total Distance and speed data
	plot_data(df)
	
main()



