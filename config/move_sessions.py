import os
import sys
import pickle
import argparse
import shutil

def move_file(src, dst, max_size):
    """Move file from src to dst"""
    print(f'Moving file: {args.src}')
    # check if src file exists
    if not os.path.exists(src):
        sys.exit(f'ERROR - Source file does not exist: {src}')
    print(f'Target directory: {args.dst}')
    # check if dst folder exists
    if not os.path.exists(dst):
        sys.exit(f'Destination folder does not exist: {dst}')
    # get last part of src and take out extension
    src_split = src.split(os.sep)[-1].split('.')[0].split('_')
    date = src_split[0]
    monkey = src_split[1].lower()
    print('  date:', date)
    print('  monkey:', monkey)
    # create new folder in dst
    new_folder = os.path.join(dst, f'{monkey}_20{date}')
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
        print('Created new folder:', new_folder)
    # move file to new folder
    tgt_file = os.path.join(new_folder, src.split(os.sep)[-1])
    size_src = os.path.getsize(src) / 1e6
    print(f'  File size: {size_src:.2f} MB')
    if size_src > max_size:
        sys.exit(f'File size exceeds maximum size: {max_size} MB')
    shutil.copy2(src, tgt_file)
    print('Done.')

def move_folder(src, dst, max_size):
    """Move folder from src to dst"""
    print(f'Moving folder: {src}')
    print(f'Target directory: {dst}')
    # check if dst folder exists
    if not os.path.exists(dst):
        sys.exit(f'Destination folder does not exist: {dst}')
    # get last part of src and take out extension
    src_split = src.split(os.sep)[-1].split('_')
    date = src_split[0]
    monkey = src_split[1].lower()
    print('  date:', date)
    print('  monkey:', monkey)
    # create new folder in dst
    new_folder = os.path.join(dst, f'{monkey}_20{date}')
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
        print('Created new folder:', new_folder)
    # move file to new folder
    tgt_folder = os.path.join(new_folder, src.split(os.sep)[-1])
    shutil.copytree(src, tgt_folder)
    print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Move video files from one folder to another')
    parser.add_argument('src', help='source folder')
    parser.add_argument('dst', help='destination folder')
    parser.add_argument('--file', help='move file', action='store_true')
    parser.add_argument('--folder', help='move folder', action='store_true')
    parser.add_argument('--max_size', help='maximum size of file to move (in MB)', default=1000, type=float)
    
    # parse arguments
    args = parser.parse_args()

    if args.src is None:
        sys.exit('Source folder not supplied')

    if args.dst is None:
        args.dst = '/mnt/c/Users/L6_00/SynologyDrive/Rahim/'
    if args.file:
        move_file(args.src, args.dst, args.max_size)
    if args.folder:
        move_folder(args.src, args.dst, args.max_size)