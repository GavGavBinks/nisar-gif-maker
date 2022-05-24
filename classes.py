from shapely.geometry import Point, Polygon
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, gridspec
from matplotlib.colors import ListedColormap, Normalize
import sys
import os
import datetime

class Footprint:
    # sensor footprint. Intended to be "attached" to satellite object
    def __init__(self, node=None ,minrange=423, swath=242, resolution=100, widthratio=3):
        self.node = node
        self.resolution = resolution
        self.minrange = minrange
        self.swath = swath
        self.width = swath/widthratio
    def get_geometry(self, step=0, debug=False):
        # use node ephemeris to find footprint geometry
        re = 6378
        latrate, lonrate = self.node.rate.loc[step]
        lon, lat = self.node.position
        lonrate = lonrate*np.cos(np.radians(lat))
        sightvec = complex(latrate, lonrate)/np.sqrt(lonrate**2+latrate**2)
        # sightvec = complex(sightvec.real / np.cos(np.radians(lat)), sightvec.imag)
        if latrate > 0:
            sightvec = complex(-1*abs(sightvec.real), -1*abs(sightvec.imag))
        else:

            sightvec = complex(abs(sightvec.real), -1*abs(sightvec.imag))
        a = self.swath/2
        b = self.width/2
        centerrange = self.minrange+a
        points = []
        for theta in np.linspace(0, 2*np.pi, self.resolution):
            points.append((centerrange+a*np.cos(theta), b*np.sin(theta)))


        points = [sightvec*complex(point[0], point[1]) for point in points]
        points = [(point.real, point.imag) for point in points]
        points = [[np.rad2deg(x/re) for x in point] for point in points]
        points = [(lon+point[0], lat+point[1]) for point in points]
        self.geometry = Polygon(points)
        x, y = self.geometry.boundary.xy

        points = [(x[i]+(x[i]-lon)/np.cos(np.radians(y[i])), y[i]) for i in range(self.resolution)]

        self.geometry = Polygon(points)
        if debug:
            lats = [point[1] for point in points]
            lons = [point[0] for point in points]
            plt.plot(lons, lats)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show()

        start = points[0]
        end = points[51]
        return self.geometry, start, end
class Frame:
    # houses plotting methods
    def __init__(self, node=False):
        self.nodes = []
        # self.add_node(node)
        self.nsteps = 0
        self.timevec = None
        self.background = World()
        map = cm.get_cmap('Greens')
        self.recencymap = ListedColormap(map(np.linspace(0, 1, 1000)))
        self.colors = [[[y for y in x]] for x in self.recencymap.colors]

        if node:
            self.add_node(node)

    # add satellite to frame
    def add_node(self, node, inherit_timevec=True):

        self.nodes.append(node)
        if not self.timevec and inherit_timevec:
            self.timevec = node.timevec
            self.nsteps = len(self.timevec)
    # make map for each timestep
    def plot_over_time(self, outputdir='C:/Users/gavin/PycharmProjects/nasa/testing'):
        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
        fovs = []
        point = []
        recency = []
        fig = plt.figure(figsize=(16, 9), constrained_layout=True)
        grid = gridspec.GridSpec(10, 1, figure=fig)
        stephistory = [x/60/60*self.nodes[0].timestep for x in range(len(self.timevec))]
        for step in self.timevec.index:
            # originalstep = step
            # step = self.timevec.index[step]

            powerax = fig.add_subplot(grid[-3,0], xlim=(0, stephistory[-1]), ylim=(0,100))
            powerax.set_title('Battery Charge')
            dataax = fig.add_subplot(grid[-2,0], xlim=(0, stephistory[-1]), ylim=(0,9))
            dataax.set_title('Data Storage (Tb)')
            downlinkax = fig.add_subplot(grid[-1,0], xlim=(0, stephistory[-1]), ylim=(0,220))
            downlinkax.set_title('Mission Data Downlinked (Tb)')
            ax = fig.add_subplot(grid[:-3,0], xlim=(-180, 180), ylim=(-90, 90), anchor='NW')
            ax.set_title('Hours Since Last Access')
            # ax = fig.add_axes([0, 0, 1, 1], aspect='equal', anchor='S', xlim=(-180, 180), ylim=(-90, 90))
            vmin = 0
            vmax = 100
            for node in self.nodes:
                for p in point:
                    point.remove(p)
                point = node.get_position(step)
                point = gpd.GeoSeries(point)
                point = point.plot(ax=ax, markersize=2, color='orange')
                point = [point]
                fov, start, end = node.get_fov(step)
                if node.active:
                    if len(fovs) and step - recency[-1] <= self.nodes[0].timestep//10:

                        # recency.append(step)
                        oldend = [x[51] for x in fovs[-1].boundary.xy]

                        oldstart = [x[0] for x in fovs[-1].boundary.xy]
                        if abs(start[0]-oldstart[0]) < 100:
                            # pass
                            # fovs[-1] = Polygon((end, oldend, oldstart, start)).union(fovs[-1]).difference(fov)
                            fovs.append(Polygon((end, oldend, oldstart, start)))
                            recency.append(step)

                    fovs.append(fov)
                    recency.append(step)

                fov = gpd.GeoSeries(fov)
                fov = fov.plot(ax=ax)
                thisrecency = [(step-x)/360 for x in recency]
                fovsframe = gpd.GeoDataFrame({'geometry':fovs, 'recency': thisrecency})
                fovsframe.plot(ax=ax, column='recency', cmap='GnBu_r', vmin=vmin, vmax=vmax)
                # for x, row in fovsframe.iterrows():
                #     fovsframe.iloc[x].plot(ax=ax, color=self.colors[row.recency])
            thishistory = [stephistory[x] for x in range(len(self.nodes[0].powerhistory))]
            norm = Normalize(vmin=vmin, vmax=vmax)
            sm = cm.ScalarMappable(norm, cmap='GnBu_r')
            cax = plt.axes([.65, .4, .075, .59])
            plt.colorbar(sm, cax=cax)
            self.nodes[0].downlink_series.boundary.plot(ax=ax, color='orange')
            self.background.background.boundary.plot(ax=ax, color='black')
            # ax.legend()
            powerax.plot(thishistory, self.nodes[0].powerhistory)
            dataax.plot(thishistory, self.nodes[0].datahistory)
            downlinkax.plot(thishistory, self.nodes[0].downlinkhistory)
            downlinkax.set_xlabel('Epoch Hour')
            plt.savefig(outputdir+'/'+str(step)+'.jpg')
            plt.clf()
            print(step)
            if False:
                if step > 300:
                    break

# originally intended to make multiple options for background map. In retrospect, probably could have been part of Frame class
class World:
    def __init__(self):
        self.background = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))




class Satellite:
    def __init__(self, inc=98.4, alt=747, file='C:/Users/gavin/PycharmProjects/nasa/nisar.csv', timestep=30):
        self.timestep = int(timestep)
        self.fov = None
        self.inc = inc
        self.alt = alt
        self.lat = []
        self.powerhistory = []
        self.lon = []
        self.ephemeris_file = file
        self.capacity = 9
        self.activity = []
        self.dutycycle = 0
        self.data = 0
        self.datahistory = []
        self.downlinkhistory = []
        self.datarate = 26/3600/24
        # self.rate = 26/3600/24*10
        self.active = True
        self.downlinkrate = .005
        self.soc = 50
        self.consumption = 100/3600
        self.restoration = self.consumption/2
        self.read_ephemeris_file()
        self.world = World()
        # calculated manually from elevation angle and NISAR altitude
        self.downlink_eca = 21.9
        self.data_downlinked = 0
        self.stations = [(15, 78),
                         (-147, 64),
                         (-70, -53)]
        self.get_downlink_polygons()
    # apply manually-calculated ECA for each site, expand based on latitude
    def get_downlink_polygons(self):

        self.downlink_polygons = Polygon()
        for lon, lat in self.stations:
            points = []
            for theta in np.linspace(0, np.pi*2, 100):
                # radius = self.downlink_eca/np.cos(np.radians(lat))
                pointlat = lat+self.downlink_eca*np.cos(theta)

                radius = self.downlink_eca/np.cos(np.radians(pointlat))
                pointlon = lon+radius*np.sin(theta)

                points.append((pointlon, pointlat))
        # Sometimes the first and last points are slightly different if the radius is large, causing the polygon not to close
            points[-1] == points[0]
            # "mirror" downlink polygon to make it appear top wrap around the earth
            poly = Polygon(points).buffer(0)
            poly = poly.union(Polygon([(lon+360, lat) for lon, lat in points]).buffer(0))
            poly = poly.union(Polygon([(lon-360, lat) for lon, lat in points]).buffer(0))
            self.downlink_polygons = self.downlink_polygons.union(poly)
        self.downlink_series = gpd.GeoSeries(self.downlink_polygons)
    def get_position(self, step=0):
        self.position = self.lon[step], self.lat[step]
        self.point = Point(self.position)
        return self.point
    def get_fov(self, step=0):
        self.step = step
        geometry, start, end = self.footprint.get_geometry(step)
        self.check_activity()
        if not self.active:
            return None, None, None
        return geometry, start, end

    def check_activity(self):
        '''
        Check downlink based on subsatellite point and downlink polygons
        Check SAR payload activity based on battery, data storage, footpring geometry

        '''
        self.check_access()
        self.check_data()
        self.active = self.storage and self.access and self.soc>0
        if self.downlink:
            if self.data < self.downlinkrate*self.timestep:
                self.data_downlinked += self.data
                self.data = 0
            else:
                self.data_downlinked += self.downlinkrate*self.timestep
                self.data -= self.downlinkrate*self.timestep
        if self.active:
            self.soc -= self.consumption*self.timestep
            self.data += self.datarate*self.timestep
        self.soc += self.restoration*self.timestep
        if self.soc > 100:
            self.soc = 100
        self.powerhistory.append(self.soc)
        self.datahistory.append(self.data)
        self.downlinkhistory.append(self.data_downlinked)
        self.activity.append(self.active)
        self.dutycycle = sum(self.activity)/len(self.activity)
        print(self.dutycycle)
        print(self.soc)
    def check_access(self):
        '''
        does footprint overlap with land?
        '''
        self.access = any([x.intersects(self.footprint.geometry) for x in self.world.background.geometry])
    def check_data(self):
        '''
        data storage less than capacity?
        subsatellite point in downlink polygons?
        '''
        self.storage = self.data < self.capacity
        self.downlink = self.downlink_polygons.contains(self.point)

    def read_ephemeris_file(self, n_checks=5, format= '%H:%M:%S.%f'):
        '''
        read ephemeris file generated102938
         1in STK
        :param n_checks: timesteps to check for timestep
        :param format: timestring format
        :return:
        '''
        data = pd.read_csv(self.ephemeris_file)
        time_to_check = [x.split(' ')[-1] for x in data['Time (UTCG)'][:n_checks]]
        time_to_check = [datetime.datetime.strptime(x, format) for x in time_to_check]
        time_to_check = [time_to_check[i+1]-x for i, x in enumerate(time_to_check[:-1])]
        time_to_check = set(time_to_check)
        if len(time_to_check) > 1:
            print('nonuniform timestep detected in ephemeris file')
            raise NotImplementedError
        else:
            ephemeris_timestep = time_to_check.pop().seconds
        data = data.iloc[::int(self.timestep/ephemeris_timestep), :]
        self.timevec = data['Time (UTCG)']
        self.lat = data['Lat (deg)']
        self.lon = data['Lon (deg)']
        self.rate = data[data.columns[data.columns.isin(['Lon Rate (deg/sec)', 'Lat Rate (deg/sec)'])]]


    def add_footprint(self, footprint=False):
        if not footprint:
            self.footprint = Footprint(self)

if __name__ == '__main__':
    if len(sys.argv) > 1:

        outputdir = sys.argv[1]
        nisar = Satellite(timestep=sys.argv[2])
        # nisar.demo()
        nisar.add_footprint()
        frame = Frame(nisar)
        # frame.add_node(nisar)
        frame.plot_over_time(outputdir=outputdir)
    else:
        nisar = Satellite()
        # nisar.demo()
        nisar.add_footprint()
        frame = Frame(nisar)
        # frame.add_node(nisar)
        frame.plot_over_time()