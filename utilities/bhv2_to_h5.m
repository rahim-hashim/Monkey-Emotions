function convert_format(format,src_file)
%CONVERT_FORMAT rewrites NIMH ML data files in a different format.
%
%   convert_format(format)           % 'bhv2', 'bhvz', 'h5', 'mat'
%   convert_format(format,filelist)  % use a cell array for multiple filenames
%
%   Jan 30, 2021    Written by Jaewon Hwang (jaewon.hwang@nih.gov, jaewon.hwang@hotmail.com)

src_dir = '';
if ~exist('src_file','var')
    [src_file,src_dir] = uigetfile({'*.bhv2;*.bhvz;*.h5','MonkeyLogic Datafile (*.bhv2;*.bhvz;*.h5)';'*.mat','MonkeyLogic Datafile (*.mat)'}, ...
        'Select NIMH ML data Files','MultiSelect','on');
end
if ischar(src_file), src_file = {src_file}; end
nfile = length(src_file);

for m=1:nfile
    fprintf('%s (%d/%d)',src_file{m},m,nfile);
    
    [~,n] = fileparts(src_file{m});
    dest_file = [n '.' format];
    if exist(dest_file,'file'), error('The file, %s, exists already in the current directory. Move to a different directory and try again.',dest_file); end
    
    src = mlfileopen([src_dir src_file{m}]);
    varname = who(src);
    varname = mlsetdiff(varname,{'IndexPosition','FileInfo','FileIndex'});
    compress = ~strncmp(varname,'Cam',3);
    
    err = [];
    try
        dest = mlfileopen(dest_file,'w');
        nvar = length(varname);
        step = [linspace(1,nvar,10) nvar+1];
        idx = 1;
        for n = 1:nvar
            dest.write(src.read(varname{n}),varname{n},compress(n));
            while step(idx) <= n
                idx = idx + 1;
                fprintf('.');
            end
        end
        fprintf('done\n');
    catch err
    end
    dest.close();
    
    src.close();
    if ~isempty(err), rethrow(err); end
end