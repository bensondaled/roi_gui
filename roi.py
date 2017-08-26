import warnings
import numpy as np, matplotlib.pyplot as pl, matplotlib.lines as mlines
import matplotlib
from matplotlib.collections import PolyCollection
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Polygon
from matplotlib.gridspec import GridSpec
import collections, os, time
from scipy.spatial.distance import euclidean as dist
import cv2

CV_VERSION = int(cv2.__version__[0])
_mpl_backend = matplotlib.get_backend().lower()

class ROI(np.ndarray):
    """An object storing ROI information for 1 or more ROIs

    Parameters
    ----------
    mask : np.ndarray
        a 2d boolean mask where True indicates interior of ROI
    pts : list, np.ndarray
        a list of points defining the border of ROI
    shape : list, tuple
        shape of the mask

    There are 2 ways to define an ROI:
    (1) Supply a mask
    (2) Supply both pts and shape

    In either case, the ROI object automatically resolves both the mask and points

    """
    __array_priority__ = 1. #ensures that ufuncs return ROI class instead of np.ndarrays
    DTYPE = bool
    _custom_attrs = ['pts']
    _custom_attrs_slice = ['pts']
    def __new__(cls, mask=None, pts=None, shape=None):
        if np.any(mask) and np.any(pts):
            warnings.warn('Both mask and points supplied. Mask will be used by default.')
        elif np.any(pts) and shape==None:
            raise Exception('Shape is required to define using points.')

        if not mask is None:
            data = mask
        elif not pts is None:
            pts = np.asarray(pts, dtype=np.int32)
            data = np.zeros(shape, dtype=np.int32)
            if CV_VERSION == 2:
                lt = cv2.CV_AA
            elif CV_VERSION == 3:
                lt = cv2.LINE_AA
            cv2.fillConvexPoly(data, pts, (1,1,1), lineType=lt)
        else:
            raise Exception('Insufficient data supplied.')
        obj = np.asarray(data, dtype=ROI.DTYPE).view(cls)
        assert obj.ndim in [2,3]
        
        return obj
    def __init__(self, *args, **kwargs):
        self._compute_pts()

    def _compute_pts(self):
        if CV_VERSION == 2:
            findContoursResultIdx = 0
        elif CV_VERSION == 3:
            findContoursResultIdx = 1
        data = self.copy().view(np.ndarray)
        if self.ndim == 2:
            data = np.array([self])

        selfpts = []
        for r in data:
            pts = np.array(cv2.findContours(r.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)[findContoursResultIdx])
            if len(pts) > 1:
                pts = np.concatenate(pts)
            pts = pts.squeeze()
            selfpts.append(pts)
        self.pts = np.asarray(selfpts).squeeze()

    def add(self, roi):
        """Add an ROI to the ROI object

        Parameters
        ----------
        roi : pyfluo.ROI, np.ndarray
            the ROI to be added

        Returns
        -------
        ROI object containing old and new ROIs
        """
        roi = np.asarray(roi, dtype=ROI.DTYPE)
        if self.ndim == 2 and roi.ndim == 2:
            result = np.rollaxis(np.dstack([self,roi]), -1)
        elif self.ndim == 3:
            if roi.ndim == 2:
                to_add = roi[np.newaxis,...]
            elif roi.ndim == 3:
                to_add = roi
            else:
                raise Exception('Supplied ROI to add has improper dimensions.')
            result = np.concatenate([self,to_add])
        else:
            raise Exception('ROI could not be added. Dimension issues.')
        result = result.astype(ROI.DTYPE)
        result._compute_pts()
        return result

    def remove(self, idx):
        """Remove ROI from ROI object

        Parameters
        ----------
        idx : int
            idx to remove
        """
        roi = self.as3d()
        result = np.delete(roi, idx, axis=0)
        result = ROI(result)
        return result

    def show(self, ax=None, labels=False, label_kw=dict(color='gray'), **patch_kw):
        """
        Shows ROI as patches

        To show just borders, use facecolor and edgecolor keywords
        """
        roi = self.as3d()

        patch_kw['alpha'] = patch_kw.get('alpha', 0.5)
        patch_kw['facecolors'] = patch_kw.get('facecolors', pl.cm.viridis(np.linspace(0,1,len(roi))))
        patch_kw['edgecolors'] = patch_kw.get('edgecolors', pl.cm.viridis(np.linspace(0,1,len(roi))))
        patch_kw['lw'] = patch_kw.get('lw', 2)

        if ax is None:
            ax = pl.gca()

        # show patches
        coll = PolyCollection(verts=[r.pts for r in roi], **patch_kw)
        ax.add_collection(coll)

        # show labels
        if labels:
            for i,r in enumerate(roi):
                ax.annotate(str(i), np.mean(r.pts, axis=0), **label_kw)

        yshape,xshape = roi.shape[1:]
        ax.axis([0,xshape,yshape,0])

        return ax

    def as3d(self):
        """Return 3d version of object

        Useful because the object can in principle be 2d or 3d
        """
        if self.ndim == 2:
            res = np.rollaxis(np.atleast_3d(self),-1)
            res.pts = np.array([res.pts])
            return res
        else:
            return self
   
    ### Special methods
    def __array_finalize__(self, obj):
        if obj is None:
            return

        for ca in ROI._custom_attrs:
            setattr(self, ca, getattr(obj, ca, None))

    def __getslice__(self,start,stop):
        #classic bug fix
        return self.__getitem__(slice(start,stop))

    def __getitem__(self, idx):
        out = super(ROI,self).__getitem__(idx)
        if not isinstance(out, ROI):
            return out
        if self.ndim == 2: #no mods when slicing a single roi
            pass
        elif self.ndim == 3: #multiple rois: need associated pts
            if isinstance(idx, tuple):
                idx = idx[0]
            for ca in ROI._custom_attrs_slice:
                setattr(out, ca, getattr(out, ca, None)[idx])
        return out

OBJ,CB,LAB = 0,1,2
class ROIView():

    """An interface for selecting, inspecting, and modifying ROI. The interface is entirely matplotlib-based and is non-blocking.

    The ROIView object stores the selected ROI as roiview.roi, and it can be accessed at any time.

    Call the end() method when done using, so that object cleans up its temporary files and matplotlib objects.
    """

    def __init__(self, img=None, roi=None, traces=None, roi_show_kw={}, imshow_kw={}, plot_kw={}):
        """Initialize an ROIView object
        """

        self._cachename = '_roicache_' + str(time.time()) + '.npy'

        # display params
        imshow_kw['cmap'] = imshow_kw.get('cmap', pl.cm.Greys_r)
        self.imshow_kw = imshow_kw
        self.roi_show_kw = roi_show_kw
        self.roi_show_kw['alpha'] = self.roi_show_kw.get('alpha', .5)
        self.roi_cmap = self.roi_show_kw.pop('cmap', pl.cm.summer)
        self.roi_color = self.roi_show_kw.pop('color', 'cyan')
        self.plot_kw = plot_kw
        self.plot_kw['color'] = self.plot_kw.get('color', 'darkslateblue')
       
        # image initialization
        self.img = img
        if isinstance(self.img, np.ndarray) and self.img.ndim==2:
            img0 = self.img
            self.nsamples = 1
        elif isinstance(self.img, np.ndarray) and self.img.ndim==3:
            img0 = self.img[0]
            self.nsamples = len(self.img)
        imgh,imgw = img0.shape
        self.empty_trace = np.zeros(self.nsamples, dtype=float)
        self.empty_trace[:] = np.nan

        # layout params
        figw,figh = 12,7
        aximg_x,aximg_y = .3, .25
        aximg_w = .4
        aximg_h = ((imgh/imgw) * aximg_w) * (figw/figh)
        axtr_x,axtr_y = .2, .1
        axtr_w,axtr_h = .6, .1
        button_x = 0.
        button_y0 = 0.01
        button_w = .1
        button_htotal = .98 # total vertical space dedicated to buttons
        slider_x = axtr_x
        slider_y = .06
        slider_w = axtr_w
        slider_h = .04

        # fig & axes
        self.fig = pl.figure(figsize=(figw,figh))
        self.ax_img = self.fig.add_axes([   aximg_x, 
                                            aximg_y, 
                                            aximg_w, 
                                            aximg_h])
        self._im = self.ax_img.imshow(img0, **self.imshow_kw)
        self.ax_img.set_autoscale_on(False)
        self.ax_img.axis('off')
        self.ax_trace = self.fig.add_axes([ axtr_x,
                                            axtr_y,
                                            axtr_w,
                                            axtr_h])
        self.ax_trace.set_xlim([0, self.nsamples-1])
        self.ax_trace.set_xticks([])
        self.data_cursor, = self.ax_trace.plot([0,0],[0,1],color='indianred',lw=.5,linestyle=':')
        self.trace_data, = self.ax_trace.plot(self.empty_trace, **self.plot_kw)

        # define buttons; convention: name: [obj, callback, label]
        self.buts = collections.OrderedDict([   
                    ('select', [None,self.evt_select,'Select (T)']),
                    ('remove', [None,self.evt_remove,'Remove (X)']),
                    ('hideshow', [None,self.evt_hideshow,'Hide (V)']),
                    ('next', [None,self.evt_next,'Next (n/N)']),
                    ('save', [None,self.cache,'Save (cmd-S)']),
                    ('inspect', [None,self.evt_inspect,'Inspect (i)']),
                        ]) 
        # create buttons
        button_h = button_htotal / len(self.buts)
        for bi,(name,(obj,cb,lab)) in enumerate(self.buts.items()):
            ax = self.fig.add_axes([button_x,
                                    button_y0 + button_h*bi,
                                    button_w,
                                    button_h])
            ax.spines['right'].set_visible(True)
            ax.spines['top'].set_visible(True)
            but = Button(ax, lab)
            but.label.set_fontsize('x-small')
            but.on_clicked(cb)
            self.buts[name][OBJ] = but
        # create time slider
        ax = self.fig.add_axes([slider_x,
                                slider_y,
                                slider_w,
                                slider_h])
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        sli = Slider(   ax=ax,
                        label='timeslider', 
                        valmin=0, 
                        valmax=self.nsamples-1,
                        valinit=0,
                        facecolor='gray', 
                        edgecolor='none', 
                        alpha=0.5, 
                        valfmt='%0.0f'  )
        sli.label.set_visible(False)
        sli.vline.set_alpha(0.)
        sli.poly.set_alpha(0)
        sli.poly.set_visible(False)
        sli.on_changed(self.evt_slide)
        self.sli = sli
        self.time_cursor, = ax.plot([0,0], [0,1], 
                color='indianred',
                linewidth=2)
        ax.set_xticks([0, self.nsamples-1])

        # callbacks
        self.fig.canvas.mpl_connect('button_press_event', self.evt_click)
        self.fig.canvas.mpl_connect('key_press_event', self.evt_key)
        #self.fig.canvas.mpl_connect('pick_event', self.evt_pick) # this is a nice feature but implementation using highlighting is actually easier without it
        self.fig.canvas.mpl_connect('motion_notify_event', self.evt_motion)
        # remap back key
        self.original_mpl_back_key = pl.rcParams['keymap.back']
        pl.rcParams['keymap.back'] = ''
        
        # runtime vars
        self._mode = '' # select, remove
        self._hiding = False
        self._selection = []
        self._selection_patches = []
        self.cur_idx = 0
        self.cur_inspect_idx = None
        self.did_edit = False

        # init roi
        self.roi = None
        self._roi_patches = []
        self._roi_centers = []
        self.add_roi(mask=roi)

        # init traces
        self.traces = traces
        if isinstance(self.traces, np.ndarray):
            self.traces = [i for i in self.traces]
        if self.traces is not None:
            assert len(self.traces) == len(self.roi), 'Number of traces does not match number of ROI'
        elif self.traces is None and self.roi is not None:
            self.traces = [self.empty_trace for i in range(len(self.roi))]

    def set_window_position(self, x, y):
        """Needs fixing - currently *moves* window instead of positioning it
        """
        if _mpl_backend == 'tkagg':
            self.fig.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
        elif backend == 'wxagg':
            self.fig.canvas.manager.window.SetPosition((x, y))
        else:
            # you can also use window.setGeometry
            f.canvas.manager.window.move(x, y)

    def reset_mode(self):
        if self._mode == 'select':
            self.evt_select()
        elif self._mode == 'remove':
            self.evt_remove()
        elif self._mode == 'inspect':
            self.evt_inspect()
        self.update_patches()

    def evt_slide(self, value):
        self.cur_idx = int(np.round(value))
        self.set_cursor(value)
        self.set_img(self.img[self.cur_idx])

    def set_cursor(self, value):
        self.time_cursor.set_xdata([value, value])
        self.data_cursor.set_xdata([value, value])
        self.sli.valtext.set_text(str(int(np.round(value))))
        self.fig.canvas.draw()

    def get_closest_roi_idx(self, x, y):
        if self.roi is None or len(self.roi) == 0:
            return None
        best = np.argmin([dist((x,y), c) for c in self._roi_centers])
        return best

    def evt_motion(self, evt):
        if self._mode not in ['remove','inspect']:
            return

        if evt.inaxes != self.ax_img:
            self.update_patches()
            return

        x,y = evt.xdata,evt.ydata
        closest = self.get_closest_roi_idx(x, y) 

        if closest is None:
            return
        elif self._mode == 'inspect' and closest==self.cur_inspect_idx:
            return
        else:
            cols = dict(remove='red', inspect=self.plot_kw['color'])
            self.update_patches(draw=False)
            self._roi_patches[closest].set_color(cols[self._mode])
            self._roi_patches[closest].set_alpha(.9)
            self.fig.canvas.draw()

    def evt_select(self, *args):
        but,_,lab = self.buts['select']
        if self._mode == 'select':
            self._mode = ''
            but.label.set_text(lab)
            but.label.set_color('k')
        elif self._mode != 'select':
            self.reset_mode()
            self._mode = 'select'
            but.label.set_text('STOP (T)')
            but.label.set_color('red')
        self.fig.canvas.draw()
    
    def evt_remove(self, *args):
        but,_,lab = self.buts['remove']
        if self._mode == 'remove':
            self._mode = ''
            but.label.set_text(lab)
            but.label.set_color('k')
            self.update_patches()
        elif self._mode != 'remove':
            self.reset_mode()
            self._mode = 'remove'
            but.label.set_text('STOP (X)')
            but.label.set_color('red')
        self.fig.canvas.draw()

    def evt_inspect(self, *args):
        but,_,lab = self.buts['inspect']
        if self._mode in ['inspect']:
            self._mode = ''
            but.label.set_text(lab)
            but.label.set_color('k')
            self.set_trace(None)
            self.cur_inspect_idx = None
            self.update_patches()
        elif self._mode not in ['inspect']:
            self.reset_mode()
            self._mode = 'inspect'
            but.label.set_text('Inspecting')
            but.label.set_color('grey')
        self.fig.canvas.draw()

    def evt_hideshow(self, *args):
        but,_,lab = self.buts['hideshow']
        if self._hiding:
            but.label.set_text('Hide (V)')
            self._hiding = False
            for p in self._roi_patches:
                p.set_visible(True)
        elif self._hiding == False:
            but.label.set_text('Show (V)')
            self._hiding = True
            for p in self._roi_patches:
                p.set_visible(False)
        self.fig.canvas.draw()
    
    def evt_next(self, *args):
        if self.nsamples==1:
            return
        self.cur_idx += 1
        if self.cur_idx >= self.nsamples:
            self.cur_idx = self.nsamples-1
            return
        n = self.img[self.cur_idx]
        self.set_img(n)
        self.set_cursor(self.cur_idx)

        if self.did_edit:
            self.cache()
        self.did_edit = False
    
    def evt_prev(self, *args):
        if self.nsamples==1:
            return
        self.cur_idx -= 1
        if self.cur_idx < 0:
            self.cur_idx = 0
            return
        n = self.img[self.cur_idx]
        self.set_img(n)
        self.set_cursor(self.cur_idx)
        
        if self.did_edit:
            self.cache()
        self.did_edit = False

    def cache(self, *args):
        np.save(self._cachename, self.roi)

    def evt_key(self, evt):

        if evt.key == 'z':
            self.remove_roi(-1)

        elif evt.key == 'escape':
            self.reset_mode()

        elif evt.key == 't':
            self.evt_select()
        elif evt.key == 'x':
            self.evt_remove()
        elif evt.key == 'v':
            self.evt_hideshow()
        elif evt.key == 'n':
            self.evt_next()
        elif evt.key == 'N':
            self.evt_prev()
        elif evt.key == 'i':
            self.evt_inspect()
        elif evt.key == 'super+s':
            self.cache()

        if self._mode != 'select':
            return

        if evt.key == 'enter':
            self.add_roi(pts=self._selection)
            self._clear_selection()
        elif evt.key == 'backspace':
            if len(self._selection) > 0:
                self._selection = self._selection[:-1]
                self._selection_patches[-1].remove()
                self._selection_patches = self._selection_patches[:-1]
                self.fig.canvas.draw()
    
    def _clear_selection(self):
        self._selection = []
        for p in self._selection_patches:
            p.remove()
        self._selection_patches = []
        self.fig.canvas.draw()
    
    def evt_click(self, evt):
        if self._mode not in ['select', 'remove', 'inspect']:
            return
        if evt.inaxes != self.ax_img:
            return

        pt = [round(evt.xdata), round(evt.ydata)]

        if self._mode == 'remove':
            closest = self.get_closest_roi_idx(evt.xdata, evt.ydata)
            self.remove_roi(closest)

        elif self._mode == 'select':
            self._selection_patches.append(self.ax_img.plot(pt[0], pt[1], marker='+', markersize=8, color='white')[0])
            self._selection.append(pt)

        elif self._mode in ['inspect']:
            closest = self.get_closest_roi_idx(evt.xdata, evt.ydata)
            self.inspect(closest)

        self.fig.canvas.draw()

    def inspect(self, idx):
        # trace
        self.set_trace(idx)
        # roi
        self.cur_inspect_idx = idx
        self.update_patches(draw=True)

    def evt_pick(self, evt):
        """Not in use currently, but is the correct implementation for ROI removal if pick events are connected
        """
        if self._mode != 'remove':
            return
        obj = evt.artist
        idx = self._roi_patches.index(obj)
        self.remove_roi(idx)

    def remove_roi(self, idx):
        if self.roi is None or len(self.roi)==0:
            return
        self._roi_patches[idx].remove()
        self.roi = self.roi.remove(idx)
        del self._roi_patches[idx]
        del self._roi_centers[idx]
        del self.traces[idx]
        self.update_patches()
        self.did_edit = True

    def set_img(self, img):
        if self._im is None:
            self._im = self.ax_img.imshow(img, **self.imshow_kw)
        else:
            self._im.set_data(img)
        self.fig.canvas.draw()

    def set_trace(self, idx):
        if idx is None:
            tr = self.empty_trace
        else:
            tr = self.traces[idx]
        self.trace_data.set_ydata(tr)

    def add_roi(self, pts=None, mask=None):
        if pts is None and mask is None:
            return

        if pts is None and not np.any(mask):
            return

        if mask is None and len(pts)==0:
            return

        roi = ROI(pts=pts, mask=mask, shape=self._im.get_array().shape)
        if self.roi is None:
            self.roi = roi
            self.traces = []
        else:
            self.roi = self.roi.add(roi)

        # show
        roi = roi.as3d()
        for r in roi:
            poly = Polygon(r.pts, picker=5, **self.roi_show_kw)
            self.ax_img.add_patch(poly)
            self._roi_patches.append(poly)
            self._roi_centers.append(np.mean(r.pts, axis=0))
            self.traces.append(self.empty_trace)
        self.update_patches()
        self.did_edit = True

    def update_patches(self, draw=True):
        if self.roi is not None and len(self.roi) > 0:
            
            # determine color/s
            if self.roi_cmap is not None:
                cols = self.roi_cmap(np.linspace(0,1,len(self.roi)))
            if self.roi_color is not None: # note that color takes predence over cmap
                cols = [self.roi_color] * len(self._roi_patches)

            for col,p in zip(cols,self._roi_patches):
                p.set_color(col)
                p.set_linewidth(1)
                p.set_alpha(self.roi_show_kw['alpha'])
       
            if self.cur_inspect_idx is not None:
                self._roi_patches[self.cur_inspect_idx].set_edgecolor(self.plot_kw['color'])
                self._roi_patches[self.cur_inspect_idx].set_alpha(.8)
                self._roi_patches[self.cur_inspect_idx].set_facecolor('none')
                self._roi_patches[self.cur_inspect_idx].set_linewidth(3)
        if draw:
            self.fig.canvas.draw()

    def end(self):
        if os.path.exists(self._cachename):
            os.remove(self._cachename)
        pl.close(self.fig)
        pl.rcParams['keymap.back'] = self.original_mpl_back_key

if __name__ == '__main__':
    img = np.random.random([200,512,512])
    roi = np.load('testroi.npy')
    tr = np.random.random([len(roi), len(img)])

    rv = ROIView(img=img, roi=roi, traces=tr)
