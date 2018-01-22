- how to extract a subset of boost: 
	+ bcp.exe mpl type_traits utility preprocessor fusion --boost="G:\Projects\PCL 1.8.1\3rdParty\Boost\include\boost-1_64" "G:\Projects\Oh\Oh\3rdParty"
- how to build bcp
	+ download boost source
	+ build boost buid b2.exe: G:\Projects\boost_1_64_0\bootstrap.bat
	+ build bcp.exe: b2 tools/bcp
	+ https://stackoverflow.com/questions/440585/building-boost-bcp