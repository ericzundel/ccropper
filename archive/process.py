#
# Process all photos in a directory
#
# This script scans a directory for files with a specific suffix,
# assuming they are protos of circles to be sorted and cropped.
#
# This script tries to take advantage of the multi-cpu architecture
#
# Process 1
# Find text indicating the lot number in the image. If present, move the file
# into a directory named with the found text and rename it.
#
# Process 2
# Look for a circle in the middle of the photo, crop the photo and save it out.
# Perform any image processing (like white balancing) in this process


import argparse
import concurrent.futures
import os
import pprint
import shutil
import sys
import time

import numpy as np
import pytesseract
import cv2


CAMERAFSROOT="E:\\"                  # Camera filesystem default
INPUTDIR=os.path.join(CAMERAFSROOT, "DCIM", "100KFZ55")
                                   # Directory where raw camera files are held
IMAGESUFFIX=".JPG"                 # Suffix of files created by Kodak PixPro F55
OUTPUTDIR=os.path.join("C:\\","Users","Ericz","OneDrive","Pictures","circles")
                                   # Directory to keep processed files
OUTPUTSUFFIX=".png"                # Suffix for files to create
NUMTHREADS=4                       # Number of threads to use for processing
args=None

TESSERACT_CONFIG="-c load_system_dawg=false -c  load_freq_dawg=false --psm 7"

def debug(*arguments, **kwargs):
    global args
    if (args.debug):
        print(*arguments, **kwargs, file=sys.stderr, flush=True)

def warn(*arguments, **kwargs):
    print(*arguments, **kwargs, file=sys.stderr, flush=True)
        
def parse_args():
    parser = argparse.ArgumentParser(
        prog = 'process',
        description = 'Looks for files with a circle in the center of an image and crops it with a border');
    parser.add_argument("-inputdir", 
                        default=INPUTDIR,
                        help='Set the directory to read raw files from')
    parser.add_argument("-imagesuffix", "-is",
                        default=IMAGESUFFIX,
                        help='Set the suffix to scan for in the inputdir directory')
    parser.add_argument('-outputdir', '-o',
                        default=OUTPUTDIR,
                        help='Set the directory where files will be moved to')
    parser.add_argument("-outputsuffix", "-os",
                        default=OUTPUTSUFFIX,
                        help='Set the suffix to use for processed files')
    parser.add_argument("-numthreads", "-n",
                        default=NUMTHREADS,
                        help='Set the number of threads to run in parallel')
    parser.add_argument('-debug', '-d', action='store_true',
                        help='Enable debugging')
    
    args = parser.parse_args()
    if (args.debug):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(args)
        print(flush=True)
        
    return args

def read_lot_number(raw_filename):
    debug("Reading %s" % (raw_filename))
    # Read into opencv2
    img = cv2.imread(raw_filename)
    # Grab just the top left 1/3 of the image
    h,w,c = img.shape
    cropped_image_height = int(h/3)
    cropped_image_width = int(w/3)
    lot_image = np.zeros((cropped_image_height, cropped_image_width, 3),
                         np.uint8)
    lot_image[0:cropped_image_height,0:cropped_image_width] = \
        img[0:cropped_image_height,0:cropped_image_width]

    debug("Looking for lot number in cropped %s" % (raw_filename))
    text = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
    debug("Read text '%s' from file %s" % (text, raw_filename))
    
    
# Process a single thread. This happens as a part of a pool
def process_raw_file(raw_filename):
    global args
    source=raw_filename
    if (not os.path.exists(source)):
        raise FileExistsError("Couldn't find file: " + raw_filename)

    text_found = False
    lot_text = read_lot_number(raw_filename)
    if (lot_text and lot_text.strip() != None and lot_text.strip() != ""):
        lot_text_words = split(lot_text)
        if len(lot_text_words) >= 2 and lot_text_words[0] == "Lot":
               lot_number = lot_text_words(1)
               lot_text = "%s_%s" % (lot_text_words[0], lot_text_words[1])
               text_found = True

    if (text_found):
        destdir = os.path.join(args.outputdir, lot_text)
        os.mkdir(destdir)
        dest = os.path.join(destdir, "%s-%s" % (lot_text, os.path.basename(raw_filename)))
    else:
        dest=os.path.join(args.outputdir, os.path.basename(raw_filename))
    

    debug("Would move %s to %s" % (source, dest))
    result = shutil.move(source,dest)
    debug("Result of move %s to %s is %s" % (source, dest, result))
    

# Find raw image files in a directory and fork off a process to handle each one
def process_raw_directory(outputdir):
    global args
    
    files = os.listdir(args.inputdir)
    results = {}
    numworkers = int(args.numthreads)
    debug("Starting a pool with %d workers" % (numworkers))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=numworkers) as executor:
        futures={}
        for filename in files:
            filename = os.path.join(args.inputdir, filename)
            if os.path.isdir(filename):
                debug("Skipping '%s'. Directory." % (filename))
            elif (not filename.endswith(args.imagesuffix)):
                debug("Skipping '%s' . Suffix doesn't match: %s " % (filename, args.imagesuffix))
            else:
                # Process each file in a separate process
                future = executor.submit(process_raw_file, filename)
                futures[future] = filename

        debug("Waiting on %d tasks to finish" % len(futures))
            
        for future in concurrent.futures.as_completed(futures):
            filename = futures[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (filename, exc))
            else:
                print('%s processing complete!' % (filename), flush=True)
            
    
def main():
    global args
    args = parse_args()
    if (not os.path.exists(args.inputdir)):
        print("Input directory does not exist: ",
              args.inputdir, file=sys.stderr)
        sys.exit(1)
    if (not os.path.exists(args.outputdir)):
        try:
              os.mkdir(args.outputdir)
        except FileExistsError:
              pass
        except FileNotFoundError:
              print("Output directory cannot be created, FileNotFoundError: ",
                    args.outputdir, file=sys.stderr)
              sys.exit(1)             

    process_raw_directory(args.outputdir)
    print("Processing Complete!")        


if __name__ == "__main__":
    main() 

