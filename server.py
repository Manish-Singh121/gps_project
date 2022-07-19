from flask import Flask, render_template
from gps import *
import multiprocessing
import time
plt.switch_backend('agg')

app = Flask(__name__, static_url_path='', static_folder='./static', template_folder='./templates')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config["TEMPLATES_AUTO_RELOAD"] = True

count = 0
epsilon = 1/1000 #in km
trace_file_name = "./data.txt"

@app.route('/')
def home():
	#load data and filter
	df = read_data(trace_file_name)
	df = kalman_filter(df)

	#create map
	create_map(df, "static/mymap.html", "")
			
	return render_template('main_page.html')
	
@app.route('/live')
def home_live():
	#load data and filter
	df = read_data(trace_file_name)
	df = kalman_filter(df)

	#create map
	create_map(df, "static/mymap.html", "live")
			
	return render_template('live.html')
	
@app.route('/rdp')
def home_rdp():
	#load data and filter
	df = read_data(trace_file_name)
	df = kalman_filter(df)
	
	# Curve Smooth map
	df = rdp(df, epsilon)
	df.reset_index(inplace = True)

	#create map
	create_map(df, "static/rdp.html", "")
			
	return render_template('main_page_rdp.html')

@app.route('/plot')
def home_plot():
	df = read_data(trace_file_name)
	df = kalman_filter(df)
	
	# plotting Total Distance and speed data
	plot_data(df)
			
	return render_template('plot.html')
	
def update_map():
	while True:
		df = read_data(trace_file_name)
		df = kalman_filter(df)
		
		global count
		if count != len(df):
			print("updating")
			# Simple map
			# create html map
			create_map(df, "static/mymap.html")
			
			# set count for update check
			count = len(df)
			
			# Curve Smooth map
			df = rdp(df, epsilon)
			df.reset_index(inplace = True)
			create_map(df, "static/rdp.html")
		
		
		time.sleep(5.0)   
   
if __name__ == '__main__':
  #background_thread = multiprocessing.Process(target=update_map)
  #background_thread.start()
  app.run()
