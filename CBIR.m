function varargout = CBIR(varargin)
% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @CBIR_OpeningFcn, ...
    'gui_OutputFcn',  @CBIR_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before CBIR is made visible.
function CBIR_OpeningFcn(hObject, eventdata, handles, varargin)
handles.folder_name=[cd '\images'];
handles.imageDataset = load([cd '\images\dataset.mat']);
handles.numOfReturnedImages = 10+1;
% Choose default command line output for CBIR
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% --- Outputs from this function are returned to the command line.
function varargout = CBIR_OutputFcn(hObject, eventdata, handles)
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
[query_fname, query_pathname] = uigetfile('*.jpg; *.png; *.bmp');
if (query_fname ~= 0)
    query_fullpath = strcat(query_pathname, query_fname);
    imgInfo = imfinfo(query_fullpath);
    [pathstr, name, ext] = fileparts(query_fullpath); % fiparts returns char type
    if ( strcmpi(ext, '.jpg') == 1 || strcmpi(ext, '.png') == 1 || strcmpi(ext, '.bmp') == 1 )        
        queryImage = imread( fullfile( pathstr, strcat(name, ext) ) );
        % display query image
        axes(handles.axes2)
        imshow(queryImage, []);
        queryImage = imresize(queryImage, [384 256]); %make the image size standard      
        % extract query image features
        queryImageFeature=Extract_features(queryImage,imgInfo,name);
        % update global variables to be used in next stage
        handles.queryImageFeature = queryImageFeature;
        handles.img_ext = ext;
        guidata(hObject, handles);
    else
        errordlg('You have not selected the correct file type');
    end
else
    return;
end


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
%Define parameters to be used in search
numOfReturnedImgs = handles.numOfReturnedImages;
queryImageFeatureVector=handles.queryImageFeature;
dataset=handles.imageDataset.dataset;
folder_name=handles.folder_name;
img_ext=handles.img_ext;
% extract image fname from queryImage and dataset
dataset_img_names = dataset(:, end);
queryImageFeatureVector(:, end) = [];
dataset(:, end) = [];
euclidean = zeros(size(dataset, 1), 1);
% compute euclidean distance
for k = 1:size(dataset, 1)
    euclidean(k) = sqrt( sum( power( dataset(k, :) - queryImageFeatureVector, 2 ) ) );
end
% add image fnames to euclidean
euclidean = [euclidean dataset_img_names];
% sort them according to smallest distance
[sortEuclidDist,~] = sortrows(euclidean);
sortedEuclidImgs = sortEuclidDist(:, 2);
% dispaly images returned by search
figure('Name','Search results','Position',[64 112 1176 561])
for m = 2:numOfReturnedImgs
    img_name = sortedEuclidImgs(m);
    img_name = int2str(img_name);
    str_img_name = strcat(folder_name,'\',img_name,img_ext);
    returned_img = imread(str_img_name);
    subplot(3, 4, m-1);
    subimage(returned_img);
    axis('off')
end
