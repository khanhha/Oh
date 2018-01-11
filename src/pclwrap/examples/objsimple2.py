# Copyright (C) 2012, The Ohzone, Inc.
#

""" Module objsimple

This module produces functionality to read and write wavefront (.OBJ) files.

http://en.wikipedia.org/wiki/Wavefront_.obj_file

The wavefront format is quite powerfull and allows a wide variety of surfaces
to be described.

This implementation does only supports vertices and faces

"""

#import visvis as vv
import numpy as np
import time
import math

class objreader():
    
    def __init__(self, f):
        self._f = f
        #self._count = 0

        # Original vertices, normals and texture coords.
        # These are not necessarily of the same length.
        self._v = []
        self._origin = [(0.0, 0.0, 0.0)]
        self.vv = [[0.0, 0.0, 0.0]]
        self.ff = [[0]]
        self._vn = []
        self._vt = []
        #vertex slice map - <vertex no> <prev vertex> <next vertex>
        self._vsm = []
        #vertex normal dot product vertex centroid
        self._vdot = []
        
        # The faces, indices to vertex/normal/texcords arrays.
        self._fv = []
        self._fn = []
        self._ft = []
        self._face = []
    
    @classmethod 
    def read(self,fname):
        """ read(fname)
        
        This classmethod is the entry point for reading OBJ files.
        
        Parameters
        ----------
        fname : string
            The name of the file to read.
        
        """
        
        t0 = time.time()
        
        # Open file
        f = open(fname, 'rb')
        # self._f = f
        try:
            reader = objreader(f)
            while True:
                reader.readLine()

        except EOFError:
            pass
        finally:
            f.close()
        # Done
        reader.finish()
        if __debug__ : print('reading mesh took ' + str(time.time()-t0) + ' seconds')
        # print("len reader = ", len(reader.vv))
        return reader

    
    def readLine(self):
        """ The method that reads a line and processes it.
        """
        #print(self._count)
        # Read line
        line = self._f.readline().decode('ascii', 'ignore')
        if not line:
            raise EOFError()
        line = line.strip()
        
        if line.startswith('v '):
            #self._v.append( self.readTuple(line) )
            self.vv.append( self.readVTuple(line) )
        elif line.startswith('vt '):
            self._vt.append( self.readTuple(line, 3) )
        elif line.startswith('vn '):
            self._vn.append( self.readVTuple(line) )
        elif line.startswith('vdot '):
            self._vdot.append( self.readTuple(line, 18) )
        elif line.startswith('f '):
            self.readFace(line)
        elif line.startswith('#'):
      	    pass # Comment
        elif line.startswith('mtllib '):
            print('Notice reading .OBJ: material properties are ignored.')
        elif line.startswith('g ') or line.startswith('s '):
            pass # Ignore groups and smoothing groups 
        elif line.startswith('o '):
            pass # Ignore object names
        elif line.startswith('usemtl '):
            pass # Ignore material
        elif not line.strip():
            pass
        else:
            print('Notice reading .OBJ: ignoring %s command.' % line.strip())

    
    def readVTuple(self, line):
        """ Reads a tuple of numbers. e.g. vertices, normals or texture coords.
        """
        numbers = [num for num in line.split(' ') if num]
        nn = len(numbers) + 1
        return [float(num) for num in numbers[1:nn]] # [start:end-1]

    def readTuple(self, line, n=3):
        """ Reads a tuple of numbers. e.g. vertices, normals or texture coords.
        """
        #line.rstrip('\\')
        for num in line.split(' '):
            if num and str(num) != '\\':
                line = '0 0 0'
        numbers = [num for num in line.split(' ') if num]
        return [float(num) for num in numbers[1:n+1]] 

    def readFace(self, line):
        """ Each face consists of three or more sets of indices. Each set
        consists of 1, 2 or 3 indices to vertices/normals/texcords.
        """
        
        # Get parts (skip first)
        indexSets = [num for num in line.split(' ') if num][1:]
        fv = []
        ft = []
        fn = []
        face = []
        idx = 0
        for indexSet in indexSets:
            # What indices were given?
            indices = [i for i in indexSet.split('/')]
            
            # Store new set of vertex/normal/texcords.
            # If there is a single face that does not specify the texcord
            # index, the texcords are ignored. Likewise for the normals.
            fv.append( indices[0] )
            if len(indices) > 1 and indices[1]:
                ft.append( indices[1] )
            else:
                ft.append( -1 )
            if len(indices) > 2 and indices[2]:
                fn.append( indices[2] )
            else:
                fn.append( -1 )
            face.append([fv[idx], ft[idx], fn[idx]])
            idx = idx+1
        self._fv.append(np.array(fv, 'int32'))
        self._ft.append(np.array(ft, 'int32'))
        self._fn.append(np.array(fn, 'int32'))
        #self._fv.append( [ fv[0], fv[1], fv[2] ] )
        #self._ft.append( [ ft[0], ft[1], ft[2] ] )
        #self._fn.append( [ fn[0], fn[1], fn[2] ] )
        self._face.append(face)
        # Done
        return
    
    
    def _absint(self, i, ref):
        i = int(i)
        if i>0 :
            return i-1
        else:
            return ref+i

    def finish(self):
        """ Converts gathere lists to numpy arrays and creates 
        BaseMesh instance.
        """
        if True:
            #self._v = np.array(self.vv[1:], 'float32')
            self._v = self.vv[1:]
            #self._fv = np.array(self._fv, 'int32')
            #self._ft = np.array(self._ft, 'int32')
            #self._fn = np.array(self._fn, 'int32')
            self._vdot = np.array(self._vdot, 'float32')
            self._origin = np.array(self.vv[0], 'float32')
            if len(self._vt) == 0 and len(self._vn) == 0:
                del(self._face[0:])
            #print('len(self._face) = ', len(self._face))
            pass;

class objwriter():
    
    def __init__(self, f):
        self._f = f
    
    
    @classmethod
    def write(cls, fname, mesh):
        """ write(fname, mesh):
        This classmethod is the entry point for writing mesh data to OBJ.

        Parameters
        ----------
        fname : string
        The filename to write to.
        mesh : raw
        Can be np.ndarray.
        """

        # Open file
        f = open(fname, 'wb')
        try:
            writer = objwriter(f)
            writer.writeMesh(mesh)
        except EOFError:
            pass
        finally:
            f.close()

    @classmethod
    def writeVerts(cls, fname, verts):
        f = open(fname, 'wb')
        try:
            writer = objwriter(f)
            writer.writeVerts_(verts)
        except EOFError:
            pass
        finally:
            f.close()

    def writeVerts_(self, verts):
        N = len(verts)
        self.writeLine('# Wavefront OBJ file')
        self.writeLine('# Created by Khanh (a Python visualization toolkit).')
        self.writeLine('#')
        self.writeLine('')
        for i in range(N):
            self.writeTuple(verts[i], 'v')

    def writeLine(self, text):
        """ Simple writeLine function to write a line of code to the file.
        The encoding is done here, and a newline character is added.
        """
        text += '\n'
        self._f.write(text.encode('ascii'))
    
    
    def writeTuple(self, val, what):
        """ Writes a tuple of numbers (on one line).
        """
        # Limit to three values. so RGBA data drops the alpha channel
        # Format can handle up to 3 texcords
        #val = val[:3]
        # Make string
        val = ' '.join([str(v) for v in val])
        # Write line
        self.writeLine('%s %s' % (what, val))



    def writeMesh(self, mesh):
        """ Write the given mesh instance.
        """
        
        # Get faces and number of vertices
        N = len(mesh._v)
        VT = len(mesh._vt)
        VN = len(mesh._vn)
        
        FV = len(mesh._fv)
        #f1 = ([ 1, -1, 2 ])
        #f2 = ([ 3, 4, -1 ])
        #f3 = ([ 5, -1, -1 ])
        #mesh._face.append([ f1, f2, f3 ])
        FF = len(mesh._face)
        
        # Write header
        self.writeLine('# Wavefront OBJ file')
        self.writeLine('# Created by visvis (a Python visualization toolkit).')
        self.writeLine('#')
        self.writeLine('')
        # Write data
        if True:
            for i in range(N):
                self.writeTuple(mesh._v[i], 'v')
            for i in range(VN):
                self.writeTuple(mesh._vn[i], 'vn')
            for i in range(VT):
                self.writeTuple(mesh._vt[i], 'vt')
            if FF > 0 :
                for i in range(FF):
                    faces = []
                    for j in range(len(mesh._face[i])):
                        for k in range(3):
                            if mesh._face[i][j][k] < 0 :
                                mesh._face[i][j][k] = ''
                        face = '/'.join([str(v) for v in mesh._face[i][j]])
                        faces.append(face)
                    #if len(mesh._face[i]) > 3:
                    #    print('faces 4444 = ', faces)
                    self.writeTuple(faces, 'f')
            else:
                for i in range(FV):
                    self.writeTuple(mesh._fv[i], 'f')
            
                

###########################
# reading dd format class
###########################

class ddreader():
    
    def __init__(self, f, n):
        self._f = f
        self._n = n
        
        # zi, r2 norm, tan(theta) = y/x, quadrant, xincr, yincr
        self._dd = []
        self._bb = []
        self._mm = []
        self._tg = []
    
    @classmethod 
    def read(cls,fname,n=6):
        """ read(fname)
        
        This classmethod is the entry point for reading OBJ files.
        
        Parameters
        ----------
        fname : string
            The name of the file to read.
        
        """
        
        t0 = time.time()
        
        # Open file
        f = open(fname, 'rb')
        try:
            reader = ddreader(f,n)
            while True:
                reader.readLine()
        except EOFError:
            pass
        finally:
            f.close()
        
        # Done
        reader.finish()
        if __debug__ : print('reading dd took ' + str(time.time()-t0) + ' seconds')
        return reader

    def getpindex(self, i, j):
        """ locating the entry index into 2D table 
        """
        imin = self._mm[0]
        istep = self._mm[2]
        imax = self._mm[1]

        jmin = self._mm[3]
        jstep = self._mm[5]
        jmax = self._mm[4]

        jsteps_per_i = int((jmax-jmin)/jstep + 0.5) + 1 
        isteps = int((imax-imin)/istep + 0.5) + 1
        pmax = isteps*jsteps_per_i - 1
        jgap = jmin - (jmax-2*math.pi)

        p0_i = int(math.floor((i-imin)/istep))

        if j-jmin < 0 :
            wrap = True
            if abs(j-jmin) <= jgap:
                p0 = p0_i*jsteps_per_i + jsteps_per_i - 1
                p1 = p0_i*jsteps_per_i + 0
            else:
                p0 = -11111 
                p1 = p0 + 1
                print ('getpindex Error:',' j: ', j,' jmin: ',jmin,' jmax: ',jmax,' jgap:', jgap)
        elif j-jmax >= 0 :
            wrap = True
            if j-jmax <= jgap:
                p0 = p0_i*jsteps_per_i + jsteps_per_i - 1
                p1 = p0_i*jsteps_per_i + 0
            else:
                p0 = -22222 
                p1 = p0 + 1
                print ('getpindex Error:',' j: ', j,' jmin: ',jmin,' jmax: ',jmax,' jgap:', jgap)
        else:
            wrap = False
            p0_offset = int(math.floor((j-jmin)/jstep))
            p0 = int(p0_i*jsteps_per_i + p0_offset)
            p1 = int(p0 + 1)
            
        p2 = int(p0 + jsteps_per_i)
        p3 = int(p1 + jsteps_per_i)

        if p2 > pmax:
            p2 = p0
        if p3 > pmax:
            p3 = p1
        

        p = [p0, p1, p2, p3, wrap]

        return p
        

    def append(self, dd2add, offset=0, mode=1):
        #mode: 0 add, mode=1 remove overlap, mode=2 replace
        # istep and jstep must be the same
        imin = self._mm[0]
        imax = self._mm[1]
        istep = self._mm[2]
        jmin = self._mm[3]
        jmax = self._mm[4]
        jstep = self._mm[5]

        iimin = dd2add._mm[0]
        iimax = dd2add._mm[1]
        iistep = dd2add._mm[2]
        jjmin = dd2add._mm[3]
        jjmax = dd2add._mm[4]
        jjstep = dd2add._mm[5]

        k2add = []
        for k in range(len(dd2add._tg)) :
            iival = iimin + dd2add._tg[k][0]
            collide = False
            for l in range(len(self._tg)) :
                ival = imin + self._tg[l][0]
                if iival == ival :
                    collide = True
            k2add.append(collide)
            #if collide == False :
            #    print ('tg add','iival: ',iival,'imin:',imin,'offset:',iival-imin)
        for k in range(len(dd2add._tg)) :
            if k2add[k] == False :
                iival = iimin + dd2add._tg[k][0]
                dd2add._tg[k][0] = iival - imin
                self._tg.append(dd2add._tg[k])
        #for k in range(len(self._tg)) :
        #    print ('tg ',self._tg[k])

        if istep != iistep and jstep != jjstep :
            print ('DD append ERROR: steps must be equal ', istep, iistep, jstep, jjstep)
            return

        iisteps = int((iimax-iimin)/iistep + 0.5) + 1

        jsteps_per_i = int((jmax-jmin)/jstep + 0.5) + 1 
        isteps = int((imax-imin)/istep + 0.5) + 1
        pmax = isteps*jsteps_per_i - 1

        print ('jmax: ', jmax, ' jmin: ',jmin,' jstep: ', jstep)
        print ('isteps: ', isteps, ' jsteps_per_i: ',jsteps_per_i,' pmax: ', pmax)

        for ii in range(0,iisteps) :
            iival = ii*iistep + iimin + offset
            i = 0
            collide = False
            while collide == False and i < isteps :
                ival = i*istep + imin
                if iival == ival :
                    collide = True
                    coll_i = i
                    coll_ii = ii
                i = i+1
            if collide == False : 
                print ('iival: ', iival, 'cur min: ', self._mm[0], 'cur max: ', self._mm[1])
                jj_offset = ii*jsteps_per_i
                dd_offset = iival - imin
                if iival > self._mm[1] :
                    self._mm[1] = iival
                elif iival < self._mm[0] :
                    self._mm[0] = iival
                for jj in range(0,jsteps_per_i) : 
                    jjindex = jj + jj_offset
                    dd2add._dd[jjindex][0] = dd_offset
                    self._dd.append(dd2add._dd[jjindex])
            elif collide == True and mode == 2 :  # now replace
                print ('True iival: ', iival, 'ival: ', ival, ' ii: ', ii, ' i: ',i)
                j_offset = coll_i*jsteps_per_i
                jj_offset = coll_ii*jsteps_per_i
                dd_offset = iival - imin
                for jj in range(0,jsteps_per_i) : 
                    jindex = jj + j_offset
                    jjindex = jj + jj_offset
                    dd2add._dd[jjindex][0] = dd_offset
                    for di in range(0,8) :
                        if di != 0 and di != 1 :
                            self._dd[jindex][di] = (self._dd[jindex][di]+dd2add._dd[jjindex][di]) / 2.0
                        elif  di == 1 and ( self._dd[jindex][di] == -1.0 or dd2add._dd[jjindex][di] == -1.0 ) :
                                break;


    
    def readLine(self):
        """ The method that reads a line and processes it.
        """
        
        # Read line
        line = self._f.readline().decode('ascii', 'ignore')
        if not line:
            raise EOFError()
        line = line.strip()
        
        if line.startswith('dd '):
            #self._vertices.append( *self.readTuple(line) )
            #self._dd.append( self.readTuple(line, 6) )
            self._dd.append( self.readTuple(line, self._n) )
        elif line.startswith('bb '):
            self._bb = self.readTuple(line, 6) 
        elif line.startswith('mm '):
            self._mm = self.readTuple(line, 6) 
        elif line.startswith('tg '):
            self._tg.append( self.readTuple(line, 9) )
        elif line.startswith('#'):
            pass # Comment
        elif not line.strip():
            pass
        else:
            print('Notice reading DD: ignoring %s command.' % line.strip())

    
    def readTuple(self, line, n=3):
        """ Reads a tuple of numbers. e.g. vertices, normals or teture coords.
        """
        numbers = [num for num in line.split(' ') if num]
        return [float(num) for num in numbers[1:n+1]]

    def finish(self):
        """ Converts gathere lists to numpy arrays and creates 
        BaseMesh instance.
        """
        if True:
            if __debug__ : print ('len1 : ', len(self._dd), len(self._tg))
            #self._dd = np.array(self._dd, 'float32')

###########################
# reading vmpa format class
###########################

class vmapreader():
    
    def __init__(self, f):
        self._f = f
        
        #vertex slice map - <vertex no> <prev vertex> <next vertex>
        self._vsm = []
        self._vsl = []
    
    @classmethod 
    def read(cls,fname):
        """ read(fname)
        
        This classmethod is the entry point for reading OBJ files.
        
        Parameters
        ----------
        fname : string
            The name of the file to read.
        
        """
        
        t0 = time.time()
        
        # Open file
        f = open(fname, 'rb')
        try:
            reader = vmapreader(f)
            while True:
                reader.readLine()
        except EOFError:
            pass
        finally:
            f.close()
        
        # Done
        reader.finish()
        if __debug__ : print('reading mesh took ' + str(time.time()-t0) + ' seconds')
        return reader

    
    def readLine(self):
        """ The method that reads a line and processes it.
        """
        
        # Read line
        line = self._f.readline().decode('ascii', 'ignore')
        if not line:
            raise EOFError()
        line = line.strip()
        
        if line.startswith('vsm '):
            self._vsm.append( self.readTuple(line, 6) )
        elif line.startswith('vsl '):
            self._vsl.append( self.readTuple(line, 6) )
        elif line.startswith('#'):
            pass # Comment
        elif not line.strip():
            pass
        else:
            print('Notice reading .vmap: ignoring %s command.' % line.strip())

    
    def readTuple(self, line, n=3):
        """ Reads a tuple of numbers. e.g. vertices, normals or teture coords.
        """
        numbers = [num for num in line.split(' ') if num]
        return [float(num) for num in numbers[1:n+1]]


    def finish(self):
        """ Converts gathere lists to numpy arrays and creates 
        BaseMesh instance.
        """
        if True:
            if __debug__ : print ('VMAP len: ', len(self._vsm), 'd: ', self._vsl)



