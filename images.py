from PIL import Image
import sys
import os
import imageio

class Gifinator:
    def __init__(self, indir, outdir, periods=[1, 5, 100, 1000], stepsizes=[5, 10, 30, 100, 600], resize=True, end=False):
        # self.initialperiod = periods[0]
        # self.intperiod = periods[1]
        self.indir = indir
        self.outdir = outdir
        self.process_initial_period(periods=periods, stepsizes=stepsizes, resize=resize, end=end)
    def process_initial_period(self, stepsizes=[5, 10, 30, 100, 600], periods=[1, 5, 100, 1000], resize=True, end=False):
        files = os.listdir(self.indir)
        files = [int(file.split('.')[0]) for file in files]
        files = sorted(files)
        step = 1
        choices = []
        while True:
            choice = sorted([(abs(file - step), file) for file in files])[0]
            if choice[0] > 100:
                break
            elif end:
                if step > end:
                    break

            choices.append(choice[1])
            step += stepsizes[len([period for period in periods if step > period])]
        for file in choices:
            image = Image.open(self.indir+'/'+str(file)+'.jpg')
            if resize:
                image = image.resize((600, 400), Image.ANTIALIAS)

            image.save(self.outdir+'/'+str(file)+'.jpeg', 'JPEG', quality=10)
        self.choices = [self.outdir+'/'+str(file)+'.jpeg' for file in choices]
    def make_gif(self):
        with imageio.get_writer('C:/Users/gavin/PycharmProjects/nasa/gif.gif', mode='I', duration=.1) as writer:
            for filename in self.choices:
                print(filename.split('/')[-1])
                image = imageio.imread(filename)

                writer.append_data(image)
        print('ook')
if __name__ == '__main__':
    target = sys.argv[1]

    destination = 'C:/Users/gavin/PycharmProjects/nasa/output'
    if 'quality' in sys.argv:
        dog = Gifinator(target, destination, resize=True, stepsizes=[5, 1, 2, 1, 5, 10, 25, 50, 100, 200, 400], periods=[100, 185, 414, 480, 800, 2000, 3000, 4000, 5000, 6000])
        dog.make_gif()
    else:
        dog = Gifinator(target, destination)
        dog.make_gif()