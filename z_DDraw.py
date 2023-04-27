# Copyright (c) 2023 Panagiotis Michalatos
# All rights reserved.

#this code is intended for educational purposes only and is not intended for commercial use
#this code is provided "as is" without warranty of any kind, either express or implied

#do not redistribute this code without permission of the author

import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import os
import torchvision.transforms as transforms
import random
import cv2
import numpy as np
import optimizer_image_utils as img_utils
import math
from typing import Tuple
from typing import List




class DPar():
    def random(min: float, max: float):
        r = random.random() * (max - min) + min
        return DPar(r, (min, max))

    def __init__(self, value: float, range : Tuple[float,float] = None):
        self.value = value
        self.range = range
        self.id : int = -1
        self.tensorID : int = -1
        self.tensor : torch.Tensor = None

    @property
    def isFixed(self) -> bool:
        return self.range == None
        
    #string representation
    def __str__(self):
        if self.isFixed:
            return f'({self.value})'
        else:
            return f'({self.range[0]}<{self.value}<{self.range[1]} : {self.tensorID})'
    
def DP(p)->DPar:
    if p is None:
        return DPar(0.0)
    if isinstance(p, DPar):
        return p
    if isinstance(p, float):
        return DPar(p)
    if isinstance(p, int):
        return DPar(float(p))
    if isinstance(p, tuple):
        return DPar(float(p[0]), (float(p[1][0]), float(p[1][1])))    


class DParameters:
    def __init__(self):
        self.params : List[DPar] = []
        self.tensor : torch.Tensor = None
        self.variableParams : List[DPar] = []
        self.fixeParams : List[DPar] = []
        self.external_tensors = []

    def add(self, p) -> DPar:
        par = DP(p)

        if par.id != -1:
            return par

        par.id = len(self.params)
        self.params.append(par)
        return par
    
    def addExternalTensor(self, tensor : torch.Tensor):
        self.external_tensors.append(tensor)

    def getOptimizableParameters(self) -> List[torch.Tensor]:
        return self.external_tensors + [self.tensor]
    
    def init(self, device, randomize: bool) -> List[torch.Tensor]:
        if self.tensor == None:

            self.fixeParams = [p for p in self.params if p.isFixed]
            self.variableParams = [p for p in self.params if not p.isFixed]

            if randomize:
                self.tensor = torch.randn(len(self.variableParams), device=device, requires_grad=True)
                for i, p in enumerate(self.variableParams):
                    p.value = self.tensor[i]
            else:
                arr = [p.value for p in self.variableParams]
                self.tensor = torch.tensor(arr, device=device, requires_grad=True)

            for i, p in enumerate(self.variableParams):
                p.tensorID = i
                p.tensor = self.tensor[i]

            for p in self.fixeParams:
                p.tensorID = -1
                p.tensor = torch.tensor(p.value, device=device, requires_grad=False)

        
        return self.external_tensors + [self.tensor]
    
    def postStep(self, use_no_grad: bool):
        if use_no_grad:
            for p in self.variableParams:
                p.tensor.clip_(p.range[0], p.range[1])
                p.value = p.tensor.item()
        else:
            with torch.no_grad():
                for p in self.variableParams:
                    p.tensor.clip_(p.range[0], p.range[1])
                    p.value = p.tensor.item()


class DParameterized:
    def __init__(self):
        pass

    def collectParameters(self, params : DParameters):
        pass

    def postStep(self):
        pass

class DColor(DParameterized):
    def random(alpha = None):
        if alpha == None:
            return DColor(DPar.random(0,1), DPar.random(0,1), DPar.random(0,1), DPar.random(0,1))
        else:
            return DColor(DPar.random(0,1), DPar.random(0,1), DPar.random(0,1), alpha)

    def __init__(self, r, g, b, a):
        self.r : DPar = DP(r)
        self.g : DPar = DP(g)
        self.b : DPar = DP(b)
        self.a : DPar = DP(a)

    def collectParameters(self, params : DParameters):
        self.r = params.add(self.r)
        self.g = params.add(self.g)
        self.b = params.add(self.b)
        self.a = params.add(self.a)
    
    @property
    def rgb(self) -> torch.Tensor:
        return torch.stack([self.r.tensor, self.g.tensor, self.b.tensor], dim=0)
    
    @property
    def rgba(self) -> torch.Tensor:
        return torch.stack([self.r.tensor, self.g.tensor, self.b.tensor, self.a.tensor], dim=0)

    #string representation
    def __str__(self):
        return f'({self.r}, {self.g}, {self.b}, {self.a})'

def Drgb(r, g, b, a=1.0) -> DColor:
    return DColor((r, (0,1)), (g, (0,1)), (b, (0,1)), a)

def Drgba(r, g, b, a) -> DColor:
    return DColor((r, (0,1)), (g, (0,1)), (b, (0,1)), (a, (0,1)))

def Dgray(gray, a=1.0) -> DColor:
    g = DPar(gray, (0,1))
    return DColor(g, g, g, a)

class DPoint(DParameterized):
    def random():
        return DPoint(DPar.random(0,1), DPar.random(0,1))
    
    def __init__(self, x, y):
        self.x : DPar = DP(x)
        self.y : DPar = DP(y)

    def collectParameters(self, params : DParameters):
        self.x = params.add(self.x)
        self.y = params.add(self.y)

    @property
    def xy(self) -> torch.Tensor:
        return torch.stack([self.x.tensor, self.y.tensor], dim=0)

    def __str__(self):
        return f'({self.x}, {self.y})'
    
class DBuffer:
    def __init__(
            self, 
            width: int, height: int, channels : int, device, *,
            initial_values_path: str = None, initial_values: List[float] = None,
            requires_grad: bool=False
            ):
        self.width = width
        self.height = height
        self.channels = channels
        self.device = device
        self.requires_grad = requires_grad

        if initial_values_path != None:
            self.tensor = img_utils.loadImageTensor(initial_values_path, channels, device, size=(width, height), normalizeImage=False, requires_grad=requires_grad)
        elif initial_values != None:
            value = torch.tensor(initial_values)
            value_grid = value.view(channels, 1, 1).repeat(1, height, width)
            self.tensor = value_grid.clone().detach().to(device).requires_grad_(requires_grad) #torch.tensor(value_grid, device=device, requires_grad=requires_grad)
        else:
            self.tensor = torch.zeros(channels, height, width, device=device, requires_grad=requires_grad)

    def clear(self):
        self.tensor.detach_().zero_()
        #self.tensor = torch.zeros(self.channels, self.height, self.width, device=self.device, requires_grad=self.requires_grad)

    def saveToPng(self, path, reverseIamgenetNormalization):
        img_utils.saveImageTensor(self.tensor, path, reverseIamgenetNormalization)

class DCanvasLayer:
    def __init__(self, width, height, device, requires_grad=False):
        self.width = width
        self.height = height
        self.rgb = DBuffer(width, height, 3,  device, requires_grad= requires_grad)
        self.alpha = DBuffer(width, height, 1,  device, requires_grad= requires_grad)

    def clear(self):
        self.rgb.clear()
        self.alpha.clear()

    def DFill(self, color: DColor, excludeAlpha):
        self.rgb.tensor[0] = color.r.tensor
        self.rgb.tensor[1] = color.g.tensor
        self.rgb.tensor[2] = color.b.tensor
        if not excludeAlpha:
            self.alpha.tensor[0] = color.a.tensor

class DCanvas:
    def __init__(self, width, height, device, *, requires_grad=False):
        self.device = device
        self.requires_grad = requires_grad
        self.width = width
        self.height = height
        self.layers : List[DCanvasLayer] = []
        self.rgb = DBuffer(width, height, 3, device, requires_grad=requires_grad)

    def clear(self):
        self.rgb.clear()
        for l in self.layers:
            l.clear()

    def fill(self, rgb, alpha):
        for l in self.layers:
            l.fill(rgb, alpha)

    def addLayer(self) -> DCanvasLayer:
        newLayer = DCanvasLayer(self.width, self.height, self.device, self.requires_grad)
        self.layers.append(newLayer)
        return newLayer        

    def computeComposite(self) -> torch.Tensor:
        for l in self.layers:
            alpha = torch.clamp(l.alpha.tensor, 0, 1)
            self.rgb.tensor = torch.lerp(self.rgb.tensor, l.rgb.tensor, alpha)
        return self.rgb.tensor

class DGrid:
    def __init__(self, width, height, device):
        self.width = width
        self.height = height
        self.device = device
        xs = torch.linspace(0.0, 1.0, steps=width, device=device)
        ys = torch.linspace(0.0, 1.0, steps=height, device=device)
        self.x, self.y = torch.meshgrid(xs, ys, indexing='xy')

    @property
    def xy(self) -> torch.Tensor:
        return torch.stack([self.x, self.y], dim=0)
    
class DShape(DParameterized):
    def __init__(self, color: DColor):
        self.color = color

    def collectParameters(self, params : DParameters):
        self.color.collectParameters(params)

    def draw(self, layer: DCanvasLayer, grid: DGrid):
        pass

class DShapeLayer(DParameterized):
    def __init__(self, layer: DCanvasLayer = None):
        self.layer = layer

    def collectParameters(self, params : DParameters):
        pass

    def draw(self, grid: DGrid):
        pass

class DShapeGroup(DShapeLayer):
    def __init__(self, shapes: List[DShape] = [], layer: DCanvasLayer = None):
        super().__init__(layer)
        self.shapes = shapes       

    def add(self, shape: DShape):
        self.shapes.append(shape)

    def collectParameters(self, params : DParameters):
        for s in self.shapes:
            s.collectParameters(params)

    def draw(self, grid: DGrid):
        for s in self.shapes:
            s.draw(self.layer, grid)

class DBlob(DShape):
    def __init__(self, center: DPoint, radius, strength, color: DColor):
        super().__init__(color)
        self.center = center
        self.radius = DP(radius)
        self.strength = DP(strength)

    def collectParameters(self, params : DParameters):
        super().collectParameters(params)
        self.center.collectParameters(params)
        self.radius = params.add(self.radius)
        self.strength = params.add(self.strength)

    #string representation
    def __str__(self):
        return f'(xy={self.center}, r={self.radius}, s={self.strength}, rgba={self.color})'
    
    def draw(self, layer: DCanvasLayer, grid: DGrid):
        d2 = (grid.x - self.center.x.tensor)**2 + (grid.y - self.center.y.tensor)**2
        s = torch.exp(-(self.radius.tensor**2)*d2) * self.strength.tensor
        rgbs = torch.stack([s*self.color.r.tensor, s*self.color.g.tensor, s*self.color.b.tensor], dim=0)
        layer.rgb.tensor = layer.rgb.tensor + rgbs
        layer.alpha.tensor = layer.alpha.tensor + s*self.color.a.tensor

class Dsdf(DParameterized):
    def __init__(self):
        super().__init__()

    def eval(self, grid: DGrid) -> torch.Tensor:
        pass

class DsdfCircle(Dsdf):
    def __init__(self, center: DPoint, radius):
        super().__init__()
        self.center = center
        self.radius = DP(radius)

    def collectParameters(self, params : DParameters):
        super().collectParameters(params)
        self.center.collectParameters(params)
        self.radius = params.add(self.radius)

    def eval(self, grid: DGrid) -> torch.Tensor:
        d2 = (grid.x - self.center.x.tensor)**2 + (grid.y - self.center.y.tensor)**2
        return torch.sqrt(d2) - self.radius.tensor

def Ddot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[0]*b[0] + a[1]*b[1] #torch.sum(a*b, dim=0).unsqueeze(0)

def Dnormsq(a: torch.Tensor) -> torch.Tensor:
    return a[0]*a[0] + a[1]*a[1] #torch.sum(a*b, dim=0).unsqueeze(0)

def Dcross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[0]*b[1] - a[1]*b[0] 

class DsdfLine(Dsdf):
    def __init__(self, p1: DPoint, p2: DPoint, width):
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        self.width = DP(width)

    def collectParameters(self, params : DParameters):
        super().collectParameters(params)
        self.p1.collectParameters(params)
        self.p2.collectParameters(params)
        self.width = params.add(self.width)

    def eval(self, grid: DGrid) -> torch.Tensor:
        p1 = self.p1.xy.unsqueeze(1).unsqueeze(2)
        p2 = self.p2.xy.unsqueeze(1).unsqueeze(2)
        p = grid.xy
        pa = p - p1
        ba = p2 - p1
        h = torch.clamp(Ddot(pa,ba) / Ddot(ba, ba), 0.0, 1.0)
        d = torch.norm(pa - ba*h, dim=0)
        return d - self.width.tensor

class DsdfTriangle(Dsdf):
    def __init__(self, p0: DPoint, p1: DPoint, p2: DPoint):
        super().__init__()
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2

    def collectParameters(self, params : DParameters):
        super().collectParameters(params)
        self.p0.collectParameters(params)
        self.p1.collectParameters(params)
        self.p2.collectParameters(params)

    def eval(self, grid: DGrid) -> torch.Tensor:
        p0 = self.p0.xy.unsqueeze(1).unsqueeze(2)
        p1 = self.p1.xy.unsqueeze(1).unsqueeze(2)
        p2 = self.p2.xy.unsqueeze(1).unsqueeze(2)
        p = grid.xy

        e0 = p1 - p0
        e1 = p2 - p1
        e2 = p0 - p2

        v0 = p - p0
        v1 = p - p1
        v2 = p - p2

        pq0 = v0 - e0 * torch.clamp(Ddot(v0, e0) / Ddot(e0, e0), 0.0, 1.0)
        pq1 = v1 - e1 * torch.clamp(Ddot(v1, e1) / Ddot(e1, e1), 0.0, 1.0)
        pq2 = v2 - e2 * torch.clamp(Ddot(v2, e2) / Ddot(e2, e2), 0.0, 1.0)

        s = torch.sign(e0[0]*e2[1] - e0[1]*e2[0])

        x0 = Ddot(pq0, pq0)
        x1 = Ddot(pq1, pq1)
        x2 = Ddot(pq2, pq2)

        y0 = s * (v0[0]*e0[1] - v0[1]*e0[0])
        y1 = s * (v1[0]*e1[1] - v1[1]*e1[0])
        y2 = s * (v2[0]*e2[1] - v2[1]*e2[0])

        minx = torch.min(x0, torch.min(x1, x2))
        miny = torch.min(y0, torch.min(y1, y2))

        return (-torch.sqrt(minx)*torch.sign(miny)).squeeze(0)

class DsdfBezier2(Dsdf):
    def __init__(self, p0: DPoint, p1: DPoint, p2: DPoint, width):
        super().__init__()
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.width = DP(width)

    def collectParameters(self, params : DParameters):
        super().collectParameters(params)
        self.p0.collectParameters(params)
        self.p1.collectParameters(params)
        self.p2.collectParameters(params)
        self.width = params.add(self.width)

    def eval(self, grid: DGrid) -> torch.Tensor:
        A = self.p0.xy.unsqueeze(1).unsqueeze(2)
        B = self.p1.xy.unsqueeze(1).unsqueeze(2)
        C = self.p2.xy.unsqueeze(1).unsqueeze(2)
        pos = grid.xy

        a = B - A
        b = A - 2*B + C
        c = 2*a
        d = A - pos

        kk = 1.0/Ddot(b,b)
        kx = kk*Ddot(a,b)
        ky = kk*(2.0*Ddot(a,a) + Ddot(d,b))/3.0
        kz = kk*Ddot(d,a)

        p = ky - kx*kx
        p3 = p*p*p
        q = kx*(2.0*kx*kx - 3.0*ky) + kz
        h = q*q + 4.0*p3

        res = torch.zeros_like(pos[0])
        mask = (h >= 0.0)

        h[mask] = torch.sqrt(h[mask])
        xx = (h[mask] - q[mask])/2.0
        xy = (-h[mask] - q[mask])/2.0
        uvx = torch.sign(xx)*torch.pow(torch.abs(xx), 1.0/3.0)
        uvy = torch.sign(xy)*torch.pow(torch.abs(xy), 1.0/3.0)
        t = torch.clamp(uvx + uvy - kx, 0.0, 1.0)
        rtx = d[0,mask] + t* (c[0] + t*b[0])
        rty = d[1,mask] + t* (c[1] + t*b[1])
        res[mask] = rtx**2 + rty**2

        mask = (h < 0.0)
        z = torch.sqrt(-p[mask])
        v = torch.acos(torch.clamp(q[mask]/(p[mask]*z*2.0), -1.0, 1.0))/3.0 
        m = torch.cos(v)
        n = 1.732050808*torch.sin(v)
        tx = 2.0*z*m - kx
        ty = -z*(m + n) - kx
        rtx1 = d[0, mask] + (c[0]+b[0]*tx)*tx
        rty1 = d[1, mask] + (c[1]+b[1]*tx)*tx
        rtx2 = d[0, mask] + (c[0]+b[0]*ty)*ty
        rty2 = d[1, mask] + (c[1]+b[1]*ty)*ty
        res[mask] = torch.min(rtx1**2 + rty1**2, rtx2**2 + rty2**2)

        return torch.sqrt(res) - self.width.tensor

class DsdfPolygon(Dsdf):
    def __init__(self, points: List[DPoint]):
        super().__init__()
        self.points = points
        
    def collectParameters(self, params : DParameters):
        super().collectParameters(params)
        for p in self.points:
            p.collectParameters(params)

    def eval(self, grid: DGrid) -> torch.Tensor:
        
        p = grid.xy
        d = Dnormsq(p - self.points[0].xy.unsqueeze(1).unsqueeze(2))
        s = torch.ones_like(d)

        j = len(self.points) - 1
        for i in range(len(self.points)):
            vi = self.points[i].xy
            vj = self.points[j].xy
            e = vj - vi
            w = p - self.points[i].xy.unsqueeze(1).unsqueeze(2)
            we = Ddot(w, e)
            ee = Ddot(e, e)
            weee = torch.clamp(we / ee, 0.0, 1.0)
            b = w - e.unsqueeze(1).unsqueeze(2) * weee.squeeze(0)
            d = torch.min(d, Dnormsq(b))
            ecw = Dcross(e, w)
            
            check1 =  p[1]>=vi[1]
            check2 =  p[1]<vj[1]
            check3 =  ecw > 0.0

            mask1 = torch.logical_and(torch.logical_and(check1, check2), check3)
            mask2 = torch.logical_and(torch.logical_and(torch.logical_not(check1), torch.logical_not(check2)), torch.logical_not(check3))
            mask = torch.logical_or(mask1, mask2)
            s[mask] = -s[mask]
       
            j = i

        return s*torch.sqrt(d)

        

class DsdfShape(DParameterized):
    def __init__(self, sdf: Dsdf, color: DColor, falloff=0.1, padding=0.1):
        super().__init__()
        self.sdf = sdf
        self.color = color
        self.falloff = DP(falloff)
        self.padding = padding
        

    def collectParameters(self, params : DParameters):
        super().collectParameters(params)
        self.sdf.collectParameters(params)
        self.color.collectParameters(params)
        self.falloff = params.add(self.falloff)

    def draw(self, layer: DCanvasLayer, grid: DGrid):
        sdf = self.sdf.eval(grid)
        rgb_dst = layer.rgb.tensor

        alpha = 1.0 - torch.clamp(sdf/self.falloff.tensor, 0.0, 1.0)
        mask = (sdf - self.padding)<0.0

        r_dst = rgb_dst[0, mask]
        g_dst = rgb_dst[1, mask]
        b_dst = rgb_dst[2, mask]
        a_dst = layer.alpha.tensor[0, mask]

        a_src = alpha[mask] * self.color.a.tensor        
 
        rgb_dst[0, mask] = torch.lerp(r_dst, self.color.r.tensor, a_src)
        rgb_dst[1, mask] = torch.lerp(g_dst, self.color.g.tensor, a_src)
        rgb_dst[2, mask] = torch.lerp(b_dst, self.color.b.tensor, a_src)
        #layer.alpha.tensor[0, mask] +=torch.clamp(a_dst + a_src, 0.0, 1.0)
        layer.alpha.tensor[0, mask] += a_src


class DsdfComposite(DShapeLayer):
    def __init__(self, sdfs: List[DsdfShape], layer: DCanvasLayer = None):
        super().__init__(layer)
        self.sdfs = sdfs

    def add(self, sdf: DsdfShape):
        self.sdfs.append(sdf)

    def addLine(self, p1: DPoint, p2: DPoint, color: DColor, thickness, falloff=0.01, padding=0.3):
        self.add(DsdfShape(DsdfLine(p1, p2, thickness), color, falloff, padding))

    def collectParameters(self, params : DParameters):
        for s in self.sdfs:
            s.collectParameters(params)

    def draw(self, grid: DGrid):
        for s in self.sdfs:
            s.draw(self.layer, grid)


class DMaskedColorLayer(DShapeLayer):
    def __init__(self,     
                width: int, height: int, device, color: DColor, *,
                alpha_filePath: str=None, alpha_needs_grad: bool = False, alpha_value: float = 1.0,    
                post_transforms = None,      
                layer: DCanvasLayer = None
        ):
        super().__init__(layer)
        self.color = color
        self.post_transforms = post_transforms
        self.alpha_needs_grad = alpha_needs_grad
        self.alpha_buffer = DBuffer(width, height, 1, device, initial_values_path=alpha_filePath, initial_values=[alpha_value], requires_grad=alpha_needs_grad)

    def collectParameters(self, params : DParameters):
        super().collectParameters(params)
        self.color.collectParameters(params)
        if self.alpha_needs_grad:
            params.addExternalTensor(self.alpha_buffer.tensor)

    def draw(self, grid: DGrid):   
        self.layer.rgb.tensor[0] = self.color.r.tensor
        self.layer.rgb.tensor[1] = self.color.g.tensor
        self.layer.rgb.tensor[2] = self.color.b.tensor
        self.layer.alpha.tensor = self.color.a.tensor * self.alpha_buffer.tensor

    def postStep(self):
        super().postStep()
        with torch.no_grad():
            if self.alpha_needs_grad:
                self.alpha_buffer.tensor.clamp_(0.0, 1.0)
                if self.post_transforms is not None:
                    transformed = self.post_transforms(self.alpha_buffer.tensor)
                    self.alpha_buffer.tensor.copy_(transformed)

    
class DImageLayer(DShapeLayer):
    def __init__(
            self,       
            width: int, height: int, device, *,   
            color_filePath: str=None,  alpha_filePath: str=None, alpha_value: float = 1.0,
            color_needs_grad: bool = False, post_transforms = None,
            layer: DCanvasLayer = None
        ):
        super().__init__(layer)
        self.color_needs_grad = color_needs_grad
        self.post_transforms = post_transforms
        self.color_buffer = DBuffer(width, height, 3, device, initial_values_path=color_filePath, requires_grad=color_needs_grad)
        self.alpha_buffer = DBuffer(width, height, 1, device, initial_values_path=alpha_filePath, initial_values=[alpha_value], requires_grad=False)

    def collectParameters(self, params : DParameters):
        super().collectParameters(params)
        if self.color_needs_grad:
            params.addExternalTensor(self.color_buffer.tensor)

    def draw(self, grid: DGrid):
        #self.layer.rgb.tensor.copy_(self.color_buffer.tensor)#*1.0
        #self.layer.alpha.tensor.copy_(self.alpha_buffer.tensor)#*1.0

        self.layer.rgb.tensor = self.color_buffer.tensor*1.0
        self.layer.alpha.tensor = self.alpha_buffer.tensor*1.0

    def postStep(self):
        super().postStep()
        with torch.no_grad():
            if self.color_needs_grad:
                self.color_buffer.tensor.clamp_(0.0, 1.0)
                #blur tensor
                #self.color_buffer.tensor = torch.nn.functional.conv2d(self.color_buffer.tensor, torch.ones(1, 1, 3, 3, device=self.color_buffer.tensor.device)/9.0, padding=1)
                #blurred = transforms.GaussianBlur(3, 0.5)(self.color_buffer.tensor)
                if self.post_transforms is not None:
                    transformed = self.post_transforms(self.color_buffer.tensor)
                    self.color_buffer.tensor.copy_(transformed)

class DDrawing(nn.Module):
    def __init__(self, width: int, height: int, device : torch.device, bgColor:DColor):
        super().__init__()
        self.width = width
        self.height = height
        self.bgColor = bgColor
        self.device = device
        self.params : DParameters = DParameters()
        self.canvas = DCanvas(width, height, device, requires_grad=False)
        self.background = self.canvas.addLayer()
        self.shapeGroups : List[DShapeLayer] = []
        self.grid = DGrid(width, height, device)

    def collectParameters(self, params: DParameters):
        self.bgColor.collectParameters(params)

    def initParameters(self, randomize=False):
        self.collectParameters(self.params)
        for s in self.shapeGroups:
            s.collectParameters(self.params)
        
        self.params.init(device=self.device, randomize=randomize)

    def addShapeGroup(self, shapes: List[DShape]) -> DShapeGroup:        
        newLayer = self.canvas.addLayer()
        newGroup = DShapeGroup(shapes, newLayer)
        self.shapeGroups.append(newGroup)
        return newGroup
    
    def addSdfGroup(self, sdfs: List[DsdfShape]) -> DsdfComposite:
        newLayer = self.canvas.addLayer()
        newGroup = DsdfComposite(sdfs, newLayer)
        self.shapeGroups.append(newGroup)
        return newGroup
    
    def addMaskedColorLayer(self, color: DColor, alpha_filePath: str=None, alpha_needs_grad: bool = False, alpha_value: float = 1.0, post_transforms=None) -> DMaskedColorLayer:
        newLayer = self.canvas.addLayer()
        newGroup = DMaskedColorLayer(
                                        self.width, self.height, self.device, color, 
                                        alpha_filePath = alpha_filePath, alpha_needs_grad = alpha_needs_grad, alpha_value = alpha_value, 
                                        post_transforms = post_transforms,
                                        layer = newLayer
                                    )
        self.shapeGroups.append(newGroup)
        return newGroup
    
    def addImageLayer(self, color_filePath: str=None, alpha_filePath: str=None, alpha_value: float = 1.0, color_needs_grad: bool = False, post_transforms=None) -> DImageLayer:
        newLayer = self.canvas.addLayer()
        newGroup = DImageLayer(
                                self.width, self.height, self.device, 
                                color_filePath = color_filePath, 
                                alpha_filePath = alpha_filePath, 
                                alpha_value = alpha_value, 
                                color_needs_grad = color_needs_grad, 
                                post_transforms = post_transforms,
                                layer = newLayer
                               )
        self.shapeGroups.append(newGroup)
        return newGroup
    

    def draw(self) -> torch.Tensor:
        self.canvas.clear()
        self.background.DFill(self.bgColor, excludeAlpha=False)
        for s in self.shapeGroups:
            s.layer.DFill(self.bgColor, excludeAlpha=True)
            s.draw(self.grid)
        return self.computeComposite()

    def computeComposite(self) -> torch.Tensor:
        return self.canvas.computeComposite()
    
    def getOptimizableParameters(self) -> List[torch.Tensor]:
        return self.params.getOptimizableParameters()
    
    def getImageTensor(self):
        return self.canvas.rgb.tensor
    
    def postStep(self, use_no_grad):
        self.params.postStep(use_no_grad)
        for s in self.shapeGroups:
            s.postStep()
    

#if running this file directly, run the following code
if __name__ == "__main__":

    rgb = torch.tensor([1.0, 2.0, 3.0])
    #make 4x4 tensor from rgb
    rgb = rgb.view(1, 3, 1, 1).repeat(1, 1, 4, 4)
    print(rgb)


    #device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    #create drawing
    bgColor = Drgb(0.9, 0.5, 0.5)
    drawing = DDrawing(224, 224, device, bgColor)

    # #create shapes
    # color1 = Drgb(0.0, 0.4, 0.9)
    # center1 = DPoint((0.5, (0,1)), (0.5, (0,1)))
    # radius1 = DPar(50.0, (0.1, 100.0))
    # strength1 = DPar(1.0, (0.1, 1))
    # blob1 = DBlob(center1, radius1, strength1, color1)

    # #create shape group
    # group1 = drawing.addShapeGroup([blob1])

    # #add 10 random blobs
    # for i in range(10):
    #     color = DColor.random(1.0)
    #     center = DPoint.random()
    #     radius = DPar.random(0.1, 20.0)
    #     strength = DPar.random(0.1, 1.0)
    #     blob = DBlob(center, radius, strength, color)
    #     group1.add(blob)

    #create sdf shapes
    # color2 = Drgb(0.0, 0.4, 0.9)
    # center2 = DPoint((0.5, (0,1)), (0.5, (0,1)))
    # radius2 = DPar(0.2, (0.1, 1.0))
    # sdf2 = DsdfCircle(center2, radius2)
    # sdfShape2 = DsdfShape(sdf2, color2)

    imgfolder = "D:/ML/Autoencoder/images/test/"

    image1 = drawing.addImageLayer(imgfolder + "n01443537_374.JPEG")

    mask1 = drawing.addMaskedColorLayer(Drgb(0.0, 1.0, 1.0), imgfolder + "n01514859_259.JPEG", alpha_needs_grad=True)

    #create sdf group
    group2 = drawing.addSdfGroup([])

    #add 10 random sdf shapes
    # for i in range(10):
    #     color = DColor.random(1.0)
    #     center = DPoint.random()
    #     radius = DPar.random(0.01, 0.2)
    #     sdf = DsdfCircle(center, radius)
    #     sdfShape = DsdfShape(sdf, color)
    #     group2.add(sdfShape)

    #add a line
    color3 = Drgb(0.0, 0.4, 0.9, 1.0)
    start = DPoint((0.1, (0,1)), (0.1, (0,1)))
    end = DPoint((0.8, (0,1)), (0.8, (0,1)))
    thickness = DPar(0.1, (0.01, 0.5))
    line = DsdfLine(start, end, thickness)
    lineShape = DsdfShape(line, color3)
    group2.add(lineShape)
    
    #add a triangle
    # color4 = Drgb(1.0, 0.4, 0.0, 1.0)
    # p1 = DPoint((0.1, (0,1)), (0.1, (0,1)))
    # p2 = DPoint((0.8, (0,1)), (0.1, (0,1)))
    # p3 = DPoint((0.5, (0,1)), (0.8, (0,1)))
    # triangle = DsdfTriangle(p1, p2, p3)
    # triangleShape = DsdfShape(triangle, color4)
    # group2.add(triangleShape)

    #add a bezier curve
    # color4 = Drgb(0.0, 1.0, 1.0, 1.0)
    # p1 = DPoint((0.1, (0,1)), (0.1, (0,1)))
    # p2 = DPoint((0.8, (0,1)), (0.1, (0,1)))
    # p3 = DPoint((0.5, (0,1)), (0.8, (0,1)))
    # thickness = DPar(0.05, (0.01, 0.5))
    # curve = DsdfBezier2(p1, p2, p3, thickness)
    # curveShape = DsdfShape(curve, color4)
    # group2.add(curveShape)

    #add a polygon
    # color5 = Drgb(0.0, 1.0, 0.0, 1.0)
    # p1 = DPoint((0.1, (0,1)), (0.1, (0,1)))
    # p2 = DPoint((0.8, (0,1)), (0.1, (0,1)))
    # p3 = DPoint((0.5, (0,1)), (0.8, (0,1)))
    # p4 = DPoint((0.5, (0,1)), (0.5, (0,1)))
    # polygon = DsdfPolygon([p1, p2, p3, p4])
    # polygonShape = DsdfShape(polygon, color5)
    # group2.add(polygonShape)

    #create parameters
    drawing.initParameters(randomize=False)

    #mask = torch.any(t.flatten(2, 3) > 0., dim=2)
    #t[mask] = 100.  # or t[mask] *= 100. for differentiability

    #draw
    drawing.draw()
    drawing.canvas.rgb.saveToPng("C:/Users/pan/Desktop/MLTest/test.png", reverseIamgenetNormalization=False)
