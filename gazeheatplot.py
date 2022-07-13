import os
import argparse
import csv
import numpy
import matplotlib
from matplotlib import pyplot, image
from glob import glob
from tqdm import tqdm


def draw_display(dispsize, imagefile=None):
   
    screen = numpy.zeros((dispsize[1], dispsize[0], 3), dtype='float32')
    # if an image location has been passed, draw the image
    if imagefile != None:
        # check if the path to the image exists
        if not os.path.isfile(imagefile):
            raise Exception("ERROR in draw_display: imagefile not found at '%s'" % imagefile)
        # load image
        img = image.imread(imagefile)

        # width and height of the image
        w, h = len(img[0]), len(img)
        # x and y position of the image on the display
        x = int(dispsize[0] / 2 - w / 2)
        y = int(dispsize[1] / 2 - h / 2)
        # draw the image on the screen
        screen[y:y + h, x:x + w, :] += img
    else:
        screen = numpy.ones((dispsize[1], dispsize[0], 3), dtype='float32')
    # dots per inch
    dpi = 100.0
    # determine the figure size in inches
    figsize = (dispsize[0] / dpi, dispsize[1] / dpi)
    # create a figure
    fig = pyplot.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = pyplot.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    # plot display
    ax.axis([0, dispsize[0], 0, dispsize[1]])
    ax.imshow(screen)  # , origin='upper')

    return fig, ax


def gaussian(x, sx, y=None, sy=None):
    
    if y == None:
        y = x
    if sy == None:
        sy = sx
    # centers
    xo = x / 2
    yo = y / 2
    # matrix of zeros
    M = numpy.zeros([y, x], dtype=float)
    # gaussian matrix
    for i in range(x):
        for j in range(y):
            M[j, i] = numpy.exp(
                -1.0 * (((float(i) - xo) ** 2 / (2 * sx * sx)) + ((float(j) - yo) ** 2 / (2 * sy * sy))))

    return M


def draw_heatmap(gazepoints, dispsize, imagefile=None, alpha=0.5, savefilename=None, gaussianwh=200, gaussiansd=None):

    fig, ax = draw_display(dispsize, imagefile=imagefile)

    # HEATMAP
    # Gaussian
    gwh = gaussianwh
    gsdwh = gwh / 6 if (gaussiansd is None) else gaussiansd
    gaus = gaussian(gwh, gsdwh)
    # matrix of zeroes
    strt = gwh // 2
    heatmapsize = int(dispsize[1] + 2 * strt), int(dispsize[0] + 2 * strt)

    heatmap = numpy.zeros(heatmapsize, dtype=float)
    # create heatmap
    for i in range(0, len(gazepoints)):
        # get x and y coordinates
        x = int(strt + gazepoints[i][0] - gwh // 2)
        y = int(strt + gazepoints[i][1] - gwh // 2)
        # correct Gaussian size if either coordinate falls outside of
        # display boundaries
        if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
            hadj = [0, gwh]
            vadj = [0, gwh]
            if 0 > x:
                hadj[0] = abs(x)
                x = 0
            elif dispsize[0] < x:
                hadj[1] = gwh - int(x - dispsize[0])
            if 0 > y:
                vadj[0] = abs(y)
                y = 0
            elif dispsize[1] < y:
                vadj[1] = gwh - int(y - dispsize[1])
            # add adjusted Gaussian to the current heatmap
            try:
                heatmap[y:y + vadj[1], x:x + hadj[1]] += gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * gazepoints[i][2]
            except:
                # fixation was probably outside of display
                pass
        else:
            # add Gaussian to the current heatmap
            heatmap[y:y + gwh, x:x + gwh] += gaus * gazepoints[i][2]
    # resize heatmap
    heatmap = heatmap[strt:dispsize[1] + strt, strt:dispsize[0] + strt]
    # remove zeros
    lowbound = numpy.mean(heatmap[heatmap > 0])
    heatmap[heatmap < lowbound] = numpy.NaN
    # draw heatmap on top of image
    ax.imshow(heatmap, cmap='jet', alpha=alpha)

    # FINISH PLOT
    # invert the y axis, as (0,0) is top left on a display
    ax.invert_yaxis()
    # save the figure if a file name was provided
    if savefilename != None:
        fig.savefig(savefilename)
    pyplot.close()
    # return fig

def gazeheatplot():
    # Required
    input_path = 'ASD'
    display_width = 1280
    display_height = 1024
    # Optional
    alpha = 0.5
    background_image = 'Images'
    ngaussian = 200
    sd = None

    if os.path.isdir(input_path):
        input_files = glob(os.path.join(input_path, '*.csv'))
        input_files.sort()

        os.makedirs(input_path + '_heatmaps', exist_ok=True)
        for input_file in tqdm(input_files):
            with open(input_file) as f:
                reader = csv.reader(f)
                next(reader, None)
                raw = list(reader)

                gaza_data = []
                if len(raw[0]) is 2:
                    gaze_data = list(map(lambda q: (int(q[1]), int(q[2]), 1), raw))
                else:
                    gaze_data = list(map(lambda q: (int(q[1]), int(q[2]), int(q[3])), raw))

                output_name = os.path.join(input_path + '_heatmaps', input_file.split('\\')[-1][:-3] + 'png')

                if background_image is not None and os.path.isdir(background_image):
                
                    filename = input_file.split('\\')[-1]
                    underscore = [i for i, letter in enumerate(filename) if letter == '_'][-1] + 1
                    bg_img = os.path.join(background_image, filename[underscore:-4] + '.png')
                else:
                    bg_img = None
                #output_name = os.path.join('heatmaps', input_file.split('\\')[-1][:-3] + 'png')
                draw_heatmap(gaze_data, (display_width, display_height), alpha=alpha, savefilename=output_name,
                            imagefile=bg_img, gaussianwh=ngaussian, gaussiansd=sd)
    else:
        with open(input_path) as f:
            reader = csv.reader(f)
            next(reader, None)
            raw = list(reader)

            gaza_data = []
            if len(raw[0]) is 2:
                gaze_data = list(map(lambda q: (int(q[1]), int(q[2]), 1), raw))
            else:
                gaze_data = list(map(lambda q: (int(q[1]), int(q[2]), int(q[3])), raw))

            if background_image is not None and os.path.isdir(background_image):
                background_image = None

            draw_heatmap(gaze_data, (display_width, display_height), alpha=alpha, savefilename=output_name,
                        imagefile=background_image, gaussianwh=ngaussian, gaussiansd=sd)

