import os
import re
import sys
import pickle
import argparse
import datetime
import shutil

MONKEY_LIST = ['gandalf', 'aragorn', 'bear']

def find_date_monkey(src):
    """find date and monkey name from src"""
    # get last part of src and take out extension
    src_split = src.split(os.sep)[-1].split('.')[0].split('_')
    monkey = [part for part in src_split if part.lower() in MONKEY_LIST][0]
    if len(monkey) == 0:
        sys.exit(f'ERROR - Monkey name not found in file name: {src}')
    # find date if YYYYMMDD or YYMMDD format and convert to YYYYMMDD
    date = [part for part in src_split if re.match(r'\d{6,8}', part)]
    if len(date) == 0:
        sys.exit(f'ERROR - Date not found in file name: {src}')
    elif len(date[0]) == 6:
        date = datetime.datetime.strptime(date[0], '%y%m%d').strftime('%Y%m%d')
    elif len(date[0]) == 8:
        date = date[0]
    return date, monkey

def copy_file(src, dst, max_size):
    """copy file from src to dst"""
    # find date and monkey name from src
    date, monkey = find_date_monkey(src)
    print('  date:', date)
    print('  monkey:', monkey)
    # create new folder in dst
    new_folder = os.path.join(dst, f'{monkey}_{date}')
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
        print('Created new folder:', new_folder)
    print(f'Copying file: {args.src}')
    print(f'Target directory: {args.dst}')
    # copy file to new folder
    tgt_file = os.path.join(new_folder, src.split(os.sep)[-1])
    # check file size
    size_src = os.path.getsize(src) / 1e6
    print(f'  File size: {size_src:.2f} MB')
    if size_src > max_size:
        sys.exit(f'File size exceeds maximum size: {max_size} MB')
    shutil.copy2(src, tgt_file)
    print('Done.')

def copy_folder(src, dst, max_size, folder_name):
    """copy folder from src to dst"""

    # find date and monkey name from src
    date, monkey = find_date_monkey(src)
    print('  date:', date)
    print('  monkey:', monkey)
    # create new folder in dst
    new_folder = os.path.join(dst, f'{monkey}_{date}')
    if folder_name is not None:
        new_folder = new_folder + '_' + folder_name
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
        print('Created new folder:', new_folder)
    print(f'Copying folder: {src}')
    print(f'Target directory: {dst}')
    # copy file to new folder
    tgt_folder = os.path.join(new_folder, src.split(os.sep)[-1])
    # check folder size
    size_src = shutil.disk_usage(src).total / 1e6
    print(f'  Folder size: {size_src:.2f} MB')
    # check each file in folder size and copy if less than max_size, print if greater
    for root, dirs, files in os.walk(src):
        for file in files:
            file_path = os.path.join(root, file)
            size_file = os.path.getsize(file_path) / 1e6
            if size_file > max_size:
                print(f'File size exceeds maximum size: {max_size} MB')
                print(f'  {file_path}: {size_file:.2f} MB')
            else:
                print(f'Copying file: {file_path}')
                shutil.copy2(file_path, tgt_folder)
    # shutil.copytree(src, tgt_folder)
    print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='copy video files from one folder to another')
    parser.add_argument('src', help='source folder')
    parser.add_argument('dst', help='destination folder')
    parser.add_argument('--file', help='copy file', action='store_true')
    parser.add_argument('--folder', help='copy folder', action='store_true')
    parser.add_argument('--max_size', help='maximum size of file to copy (in MB)', default=1000, type=float)
    parser.add_argument('--file_type', help='file type to copy', default='h5')
    parser.add_argument('--name', help='name of file or folder to copy')
    
    # parse arguments
    args = parser.parse_args()

    # check if src file/folder provided
    if args.src is None:
        sys.exit('Source folder not supplied')
    # check if src file/folder exists
    if not os.path.exists(args.src):
        sys.exit(f'ERROR - Source file does not exist: {args.src}')
    # check if dst file/folder provided
    if args.dst is None:
        args.dst = '/mnt/c/Users/L6_00/SynologyDrive/Rahim/'
    # check if dst folder exists
    if not os.path.exists(args.dst):
        sys.exit(f'ERROR - Destination folder does not exist: {args.dst}')

    # copy file or folder
    if args.file:
        copy_file(args.src, args.dst, args.max_size)
    if args.folder:
        copy_folder(args.src, args.dst, args.max_size, args.name)