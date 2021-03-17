%{
Amath 482
Winnie Shao
HW1
%}

clear; close all; clc;
load subdata.mat

% Initialization
L = 10; % spatial domain
n = 64; % Fourier modes
x2 = linspace(-L,L,n+1);
x = x2(1:n);
y =x; 
z = x;
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1];
ks = fftshift(k);
[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);

% Part 1
ave = zeros(n, n, n);
for j=1:49
    Sn(:,:,:)=reshape(subdata(:, j),n,n,n);
    Stn = fftn(Sn);
    ave = ave + Stn;
end

ave = abs(fftshift(ave));
flatAve = reshape(ave, n^3, 1);
strongest = max(flatAve);
ave = ave/strongest;

ave = ifftshift(ave);
for m = 1:n
    [j, i] = find(ave(:,:,m) == 1); 
    if isempty(i)~=1
        center_frequency = [i, j, m];
        break
    end
end
kx=k(center_frequency(1)); 
ky=k(center_frequency(2)); 
kz=k(center_frequency(3)); 

% Part 2: filter
tau = 0.2;
filter = exp(-tau*(fftshift(Kx)-kx).^2).*...
    exp(-tau*(fftshift(Ky)-ky).^2).*...
    exp(-tau*(fftshift(Kz)-kz).^2);

position = zeros(49, 3);

for realize = 1:49
    % apply filter on each realization
    Sn(:,:,:)=reshape(subdata(:, realize),n,n,n);
    Stn = fftn(Sn);
    snft = filter.* Stn;
    snf = ifftn(snft); % inverse multi-dimensional FFT
    snf = abs(snf);
    flat_snf = reshape(snf, n^3, 1);
    strongest = max(flat_snf);
    % look for the strongest signal
    for m = 1:n
        [j, i] = find(snf(:,:,m) == strongest);
        if isempty(i)~=1
            position(realize,:) = [x(i), y(j), z(m)];
            break
        end
    end

    isosurface(X,Y,Z,snf/strongest,0.8)
    axis([-L L -L L -L L]), grid on, drawnow
    xlabel('x'); ylabel('y');zlabel('z');
    pause(1)
end

plot3(position(:,1), position(:,2), position(:,3), '.-', 'MarkerSize',...
    10, 'Linewidth', 2)
axis([-L L -L L -L L]), grid on, drawnow
xlabel('x'); ylabel('y'); zlabel('z');

[kx, ky, kz]
solution = [position(49,1), position(49,2),position(49,3)];
