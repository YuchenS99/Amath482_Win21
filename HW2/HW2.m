close all; clear all; clc;

%% Part 1 (a)
[y, Fs] = audioread('GNR.m4a');
tr_gnr = length(y)/Fs; % record time in seconds
v = y';
n = length(v);
t2 = linspace(0,tr_gnr,n+1); 
t_gnr = t2(1:n); 
k_gnr = (2*pi/tr_gnr)*[0:n/2-1 -n/2:-1]; 
ks_gnr = fftshift(k_gnr);
tslide_gnr = 0:0.2:tr_gnr;
spec_gnr = zeros(length(tslide_gnr), n);
notes_gnr = zeros(1, length(tslide_gnr));

for j = 1:length(tslide_gnr)
    g = exp(-1500*(t_gnr - tslide_gnr(j)).^2);  % Use Gaussian Filter
    vgp = g .* v;
    vgpt = fft(vgp);
    [M, I] = max(vgpt);  % Get the index with strongest frequency (music score)
    
    notes_gnr(1,j) = abs(k_gnr(I))/(2*pi);
    spec_gnr(j,:) = fftshift(abs(vgpt));
end

figure(1);
pcolor(tslide_gnr, (ks_gnr/(2*pi)), spec_gnr.'),
shading interp
colormap(hot);
% plot(tslide_gnr, notes_gnr, 'o');
title ("GNR Guitar Music Score (200~900 Hz)");
xlabel('Time (s)'), ylabel('Music Notes');
ylim ([200 900]);
yticks([277.18, 311.13, 369.99, 415.30, 554.37, 698.46, 739.99]);
yticklabels({'C4#=277.18','D4#=311.13','F4#=369.99','G4#=415.30',...
    'C5#=554.37','F5=698.46','F5#=739.99'});

%% Part 2
[y, Fs] = audioread('Floyd.m4a');

v = lowpass(y', 150, Fs);
tr_f = length(v)/Fs;
n = length(v);

t2 = linspace(0,tr_f,n+1);
t_f = t2(1:n); 
k_f = (2*pi/tr_f)*[0:n/2-1 -n/2:-1]; 
ks_f = fftshift(k_f);
tslide_f = 0:0.3:tr_f;
spec_f = zeros(length(tslide_f), n);
notes_f = zeros(1, length(tslide_f));

for j = 1:length(tslide_f)
    g = exp(-1500*(t_f - tslide_f(j)).^2);  % Use Gaussian Filter
    vgp = g .* v;
    vgpt = fft(vgp);
    [M, I] = max(vgpt);  % Get the index with strongest frequency (music score)
    notes_f(1,j) = abs(k_f(I))/(2*pi);
    spec_f(j,:) = fftshift(abs(vgpt));
end

figure(2);
% pcolor(tslide_f, (ks_f/(2*pi)), spec_f.'),
% shading interp
% colormap(hot);
plot(tslide_f, notes_f,'o');
title ("Floyd Bass Music Score (0~150 Hz)");
xlabel('Time (s)'), ylabel('Music Note');
ylim([50 150])
yticks([82.41,92.49,97.99,110,123.47])
yticklabels({'E2=82.41','G2b=92.49','G2=97.99','A2=110','B2=123.47'})
hold on
one = ones(1, length(tslide_f));
plot(tslide_f, 82.41*one, 'r')
plot(tslide_f, 92.49*one, 'r')
plot(tslide_f, 97.99*one, 'r')
plot(tslide_f, 110*one, 'r')
plot(tslide_f, 123.47*one, 'r')
%% Part 3
[y, Fs] = audioread('Floyd.m4a');
v = highpass(y', 200, Fs);
tr_f = length(v)/Fs
n = length(v);
t2 = linspace(0,tr_f,n+1);
t_f = t2(1:n); 
k_f = (2*pi/tr_f)*[0:n/2-1 -n/2:-1]; 
ks_f = fftshift(k_f);
tslide_f = 0:0.3:tr_f;
spec_f = zeros(length(tslide_f), n);
notes_f = zeros(1, length(tslide_f));
for j = 1:length(tslide_f)
    g = exp(-1500*(t_f - tslide_f(j)).^2);  % Use Gaussian Filter
    vgp = g .* v;
    vgpt = fft(vgp);
    [M, I] = max(vgpt);  % Get the index with strongest frequency (music score)
    notes_f(1,j) = abs(k_f(I))/(2*pi);
    spec_f(j,:) = fftshift(abs(vgpt));
end
plot(tslide_f, notes_f,'o');
title ("Floyd Guitar Music Score (200~600 Hz)");
xlabel('Time (s)'), ylabel('Music Note');
ylim([200, 600])
yticks([220, 246.94, 293.66, 329.63, 369.99, 392, 440, 493.88, 587.33])
yticklabels({'A3=220','B3=246.94','D4=293.66','E4=329.63','F4#=369.99',...
    'G4=392','A4=440','B4=493.88','D5=587.33'})
hold on
one = ones(1, length(tslide_f));
plot(tslide_f, 220*one, 'r')
plot(tslide_f, 246.94*one, 'r')
plot(tslide_f, 293.63*one, 'r')
plot(tslide_f, 329.63*one, 'r')
plot(tslide_f, 369.99*one, 'r')
plot(tslide_f, 392*one, 'r')
plot(tslide_f, 440*one, 'r')
plot(tslide_f, 493.88*one, 'r')
plot(tslide_f, 587.33*one, 'r')