- **03-tri50.txt** : the antenna coordinates in the array format. 
Right now we are using format 3 but eventually will add a couple of columns for format 4.
Line 28 is the cable origin, at the equipment shelter, and is not an antenna.  Only type 1 are antennas.  
As you can see, x y z are in columns 1 2 7, and point type is in column 6.  

All z are set to zero now.  The antennas are above the ground by about h = 5 meters.  
We can set the heights all the same for now also, so antenna location = (x,y,z+h).  
Antenna height and antenna orientation are the two parameters that will be added in two more columns in format version 4 i.e columns 11 and 12.  



- **files3_read_formats.py**: This file will read the various formats of the antenna array layout files.  
The current one is the last one, format.  
parse_formats() is useful to use since if we make any change to the format then we can just swap out the module files3_read_formats.py.  
The commented skeleton code at the top of files3_read_formats.py should work and shows how to use parse_formats(). 
parse_formats() might also be copied and edited into parse_sources() to read the source file.  
We'll need something to read the ascii data files too, maybe parse_formats() could be edited for that too. 

- **source1.txt**: the one-source coordinates in the array format
- **source2.txt**: the double-source coordinates in the array format


We also need a data file format.  Ascii is good for this since the data is so far small and it is easy to see and work with.  Right now we have just one voltage sample per antenna!  We can make two data files, one in amplitude/phase format and the other in inphase/quadrature (cosine/sine) format.  The file names can be automatic e.g.
  source1--03-tri50--20200519-165700-000000--ap.txt
  source1--03-tri50--20200519-165700-000000--iq.txt

It's useful to have the date and time so can tell what version of the code was used.  Real data files also include dates and times so that's very realistic...  even more realistic, the last six 0's are microseconds.  

For now we can put the data values from each antenna one after the other.  Maybe later a data file for each antenna e.g.
  source1--03-tri50-1--ap--20200519-1657.txt
  source1--03-tri50-2--ap--20200519-1657.txt
  source1--03-tri50-3--ap--20200519-1657.txt
or you could do that now if you like, or have it as an option.  For real data we may have some mix of both because of the way the receivers work.

For the data above file source1--03-tri50--iq--20200519-1657.txt would be
  2.4923046977077408e-11 0.26480922790397016 # antenna 1 voltage sample
  2.4923663491911177e-11 0.0 # antenna 2 voltage sample
  2.4923352901280785e-11 0.13340556197383283 # antenna 3 voltage sample
except that we need to convert the amplitude from W to V and the phase to absolute phase instead of phase difference.  

It's good to save all plots as pdf.  The on-screen version should be optional, maybe via a parameter such as 
  screenplot = True # or False 
The plot file name can be the data file name + something e.g. 
  source1--03-tri50--20200519-165700-000000--geometry.pdf 

I've appended the latest version of the imaging data document:
imagingdata-20200515-1310.pdf
imagingdata-20200515-1310.tex

