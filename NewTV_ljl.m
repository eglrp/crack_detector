function BW = NewTV_ljl(filename, sigma, mask_sigma)
% 

% Example:
%       BW = NewTV_ljl('final_3.jpg', 5, 10);


% if nargin < 1
%     filename = '5_small.jpg';
% end

img = imread(filename);
img = img(:,:,1);
img  = 255 - img; % 将图像反转以调用TV部分的代码,因为TV会对非0的像素进行编码

im = double(img) / double(max(img(:)));

% First step is to produce the initially encode the image
% as sparse tensor tokens.
sparse_tf = calc_sparse_field(im);
% Calculate cached voting field at various angles, this way we can save
% a lot of time by preprocessing this data.
cached_vtf = create_cached_vf(sigma);
% First run of tensor voting, use ball votes weighted by
% the images grayscale.
refined_tf = calc_refined_field(sparse_tf,im,sigma);

% third run is to apply the stick tensor voting after
% zero'ing out the e2(l2) components so that everything
% is a stick vote.
[e1,e2,l1,l2] = convert_tensor_ev(refined_tf);
l2(:) = 0;
zerol2_tf = convert_tensor_ev(e1,e2,l1,l2);

T = calc_vote_stick(zerol2_tf,sigma,cached_vtf); % stick vote
[e1,e2,l1,l2] = convert_tensor_ev(T);

% 转为->(0,1)...
stick_tensor_saliency = l1 - l2;
ss = stick_tensor_saliency;
ss = (ss - min(ss(:)))/(max(ss(:)) - min(ss(:)));

last_mat = ss;
% OSTU 利用OSTU找到最大类间方差 的level
level = graythresh(last_mat);

BW = im2bw(last_mat, level);
BW = 1- BW;

% the region which are not in the mask are all 1 
% img(mask != 0) = 1;
a_mask = mask(filename, mask_sigma);
BW(a_mask == 1 ) = 1;   % BW(a_mask ~= 0) = 1; % equal?

% imshow(BW);
imwrite(BW, strcat(filename, '.jpg'), 'jpg');
end




