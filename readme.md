# cmu15468 lecture and assign
## environment 

### 1.1 raw image conversion 
1. download dcraw from [dcraw](https://www.easyhdr.com/download/dcraw/)
2. execute dcraw.exe -4 -d -v -T .\campus.nef
   output info is 
   Loading Nikon D3400 image from .\campus.nef ...
   Scaling with darkness 150, saturation 4095, and
   multipliers 2.393118 1.000000 1.223981 1.000000
   Building histograms...
   Writing data to .\campus.tiff ...
3. execute dcraw.exe -4 -D -T .\campus.nef
   generate .tiff file for process
### 1.1 python initials
