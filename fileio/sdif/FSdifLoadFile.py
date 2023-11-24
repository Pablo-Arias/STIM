from __future__ import division
from __future__ import absolute_import
import collections
import eaSDIF
import numpy as np
from six.moves import range
from six.moves import zip

def FSdifLoadFile(f0file, retrieve_nvts=False):
    '''
    Load sdif file containing a single frame/matrix type combination wit a matrix type containing
    a single row into a list with elements of the form ([times], [np.array(data)])
    where the numpy array contains the rows of matrix data and the times are the list of related frame times
    '''
    file = eaSDIF.Entity()
    frame = eaSDIF.Frame();

    try:
        res = file.OpenRead(f0file);
    except RuntimeError as err:
        res = False
    
    if res == False:
        raise IOError("FSdifLoadFile: "+f0file+" is not an sdif file or does not exist")

    if retrieve_nvts:
        NVTs = []
        for nvt in range(file.GetNbNVT()) :
            NVT = file.GetNVT(nvt)
            #NVTs become invalid when file is closed, so we need to copy the content
            NVT_deep_copy =  eaSDIF.NameValueTable()
            NVT_deep_copy.SetStreamID(NVT.GetStreamID())
            for kk in NVT:
                NVT_deep_copy.AddNameValue(kk, NVT[kk])
            NVTs.append(NVT_deep_copy)

    dlist = [];
    tlist = [];
    vec = eaSDIF.Vector()
        
    for frame in file:       
        # print frame to stdout
        # frame.Print();
        if frame.GetNbMatrix() > 0 :
            mat  = frame.GetMatrix(0);
            #    $nomat= $frame->GetMatrix(1);
            msig = mat.GetSignature();
            nrow = mat.GetNbRows();
            if nrow > 1:
                raise RuntimeError('FSdifLoadFile:: cannot read more than a signle row from '+fi0file)
            ncol = mat.GetNbCols();
            if nrow > 0  and ncol > 0 :
                mat.GetRow(vec, 0)
                dlist.append(np.array(vec));
                tlist.append(frame.GetTime() );
        
    # close file
    file.Close();
    if retrieve_nvts:
        return tlist, dlist, NVTs
    return tlist, dlist;

def FSdifStoreFile(filename, times, data, frame_type="1FQ0", matrix_type="1FQ0", dtype=float, name_value_dict=None, NVTs=None):
    '''
    save sdif file containing a single frame/matrix type combination with a matrix type containing
    a single row into a list with elements of the form ([times], [np.array(data)])
    where the numpy array contains the rows of matrix data and the times are the list of related frame times

    In the case when subsequent data entries have the same frame time those
    will be distributed over subsequent stream ids.
    '''

    esfile = eaSDIF.Entity()
    frame = eaSDIF.Frame()
    frame.SetSignature(frame_type)
    frame.SetStreamID(0)
    frame.AddMatrix(eaSDIF.Matrix())
    mdat =  frame.GetMatrix(0)
    easdif_type = eaSDIF.eFloat8
    if dtype ==  np.dtype('f4'):        
        easdif_type = eaSDIF.eFloat4
    elif dtype ==  np.dtype('i8'):
        easdif_type = eaSDIF.eInt8
    elif dtype ==  np.dtype('i4'):
        easdif_type = eaSDIF.eInt4
    elif dtype ==  np.dtype('i2'):
        easdif_type = eaSDIF.eInt2
    elif dtype ==  np.dtype('i1'):
        easdif_type = eaSDIF.eText
    elif dtype ==  np.dtype('u8'):
        easdif_type = eaSDIF.eUInt8
    elif dtype ==  np.dtype('u4'):
        easdif_type = eaSDIF.eUInt4
    elif dtype ==  np.dtype('u2'):
        easdif_type = eaSDIF.eUInt2
    elif dtype ==  np.dtype('u1'):
        easdif_type = eaSDIF.eText

    mdat.Init(matrix_type, 1, 1, easdif_type)

    nvt_stream_id = None
    if NVTs != None :
        for NVT in NVTs:
            if nvt_stream_id == None:
                nvt_stream_id = NVT.GetStreamID()
            esfile.AddNVT(NVT)

    if isinstance(name_value_dict, collections.Iterable) :
        NVT=eaSDIF.NameValueTable()
        if nvt_stream_id:
            NVT.SetStreamID(nvt_stream_id)
        for kk in name_value_dict:
            NVT.AddNameValue(kk, str(name_value_dict[kk]))
        esfile.AddNVT(NVT)

    res = esfile.OpenWrite(filename);
    if res == False:
        raise IOError("could not open file {0} for writing".format(filename))

    old_currtime = None
    stream_id    = 0
    if (not hasattr(data, 'shape')) or (len(data.shape) > 1):
        for (curtime, row) in zip(times, data) :
            # print frame to stdout
            # frame.Print();
            frame.SetTime(curtime)
            if old_currtime == None or old_currtime != curtime:
                stream_id = 0
            else:
                stream_id += 1
            old_currtime = curtime
            frame.SetStreamID(stream_id);
                
            if mdat.GetNbCols() != len(row) :
                mdat.Resize(1, len(row))
            mdat.SetRowA(row,0)
            esfile.WriteFrame(frame)
    else:
        mdat.Resize(1, 1);
        for (curtime, val) in zip(times, data):
            frame.SetTime(curtime);
            if old_currtime == None or old_currtime != curtime:
                stream_id = 0
            else:
                stream_id += 1
            old_currtime = curtime
            frame.SetStreamID(stream_id);
            mdat.Set(0,0,val);
            esfile.WriteFrame(frame)
    # close file
    esfile.Close();
