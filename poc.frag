#version 450

const vec3 c = vec3(1.,0.,-1.);
const float pi = acos(-1.),
    PHI = 1.618;

// Math

// Determine zeros of k.x*x^2+k.y*x+k.z
vec2 quadratic_zeros(vec3 k)
{
    if(k.x == 0.) return -k.z/k.y*c.xx;
    float d = k.y*k.y-4.*k.x*k.z;
    if(d<0.) return vec2(1.e4);
    return (c.xz*sqrt(d)-k.y)/(2.*k.x);
}

// Determine zeros of k.x*x^3+k.y*x^2+k.z*x+k.w
vec3 cubic_zeros(vec4 k)
{
    if(k.x == 0.) return quadratic_zeros(k.yzw).xyy;
    
    // Depress
    vec3 ai = k.yzw/k.x;
    
    //discriminant and helpers
    float tau = ai.x/3., 
        p = ai.y-tau*ai.x, 
        q = -tau*(tau*tau+p)+ai.z, 
        dis = q*q/4.+p*p*p/27.;
        
    //triple real root
    if(dis > 0.) {
        vec2 ki = -.5*q*c.xx+sqrt(dis)*c.xz, 
            ui = sign(ki)*pow(abs(ki), c.xx/3.);
        return vec3(ui.x+ui.y-tau);
    }
    
    //three distinct real roots
    float fac = sqrt(-4./3.*p), 
        arg = acos(-.5*q*sqrt(-27./p/p/p))/3.;
    return c.zxz*fac*cos(arg*c.xxx+c*pi/3.)-tau;
}

// Determine zeros of a*x^4+b*x^3+c*x^2+d*x+e
vec4 quartic_zeros(float a, float b, float cc, float d, float e) {
    if(a == 0.) return cubic_zeros(vec4(b, cc, d, e)).xyzz;
    
    // Depress
    float _b = b/a,
        _c = cc/a,
        _d = d/a,
        _e = e/a;
        
    // Helpers
    float p = (8.*_c-3.*_b*_b)/8.,
        q = (_b*_b*_b-4.*_b*_c+8.*_d)/8.,
        r = (-3.*_b*_b*_b*_b+256.*_e-64.*_b*_d+16.*_b*_b*_c)/256.;
        
    // Determine available resolvent zeros
    vec3 res = cubic_zeros(vec4(8.,8.*p,2.*p*p-8.*r,-q*q));
    
    // Find nonzero resolvent zero
    float m = res.x;
    if(m == 0.) m = res.y;
    if(m == 0.) m = res.z;
    
    // Apply Ferrari method
    return vec4(
        quadratic_zeros(vec3(1.,sqrt(2.*m),p/2.+m-q/(2.*sqrt(2.*m)))),
        quadratic_zeros(vec3(1.,-sqrt(2.*m),p/2.+m+q/(2.*sqrt(2.*m))))
    )-_b/4.;
}

// Created by David Hoskins and licensed under MIT.
// See https://www.shadertoy.com/view/4djSRW.
// float->float hash function
float hash11(float p)
{
    p = fract(p * .1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

// Created by David Hoskins and licensed under MIT.
// See https://www.shadertoy.com/view/4djSRW.
// vec2->float hash function
float hash12(vec2 p)
{
	vec3 p3  = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Created by David Hoskins and licensed under MIT.
// See https://www.shadertoy.com/view/4djSRW.
// vec3->float hash function
float hash13(vec3 p3)
{
	p3  = fract(p3 * .1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}

// Created by David Hoskins and licensed under MIT.
// See https://www.shadertoy.com/view/4djSRW.
// float->vec2 hash function
vec2 hash21(float p)
{
	vec3 p3 = fract(vec3(p) * vec3(.1031, .1030, .0973));
	p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx+p3.yz)*p3.zy);

}

// Created by David Hoskins and licensed under MIT.
// See https://www.shadertoy.com/view/4djSRW.
// vec2->vec2 hash function
vec2 hash22(vec2 p)
{
	vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx+33.33);
    return fract((p3.xx+p3.yz)*p3.zy);

}

// Created by David Hoskins and licensed under MIT.
// See https://www.shadertoy.com/view/4djSRW.
// vec3->vec2 hash function
vec2 hash23(vec3 p3)
{
	p3 = fract(p3 * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx+33.33);
    return fract((p3.xx+p3.yz)*p3.zy);
}

// Created by David Hoskins and licensed under MIT.
// See https://www.shadertoy.com/view/4djSRW.
// float->vec3 hash function
vec3 hash31(float p)
{
   vec3 p3 = fract(vec3(p) * vec3(.1031, .1030, .0973));
   p3 += dot(p3, p3.yzx+33.33);
   return fract((p3.xxy+p3.yzz)*p3.zyx); 
}

// Created by David Hoskins and licensed under MIT.
// See https://www.shadertoy.com/view/4djSRW.
// vec2->vec3 hash function
vec3 hash32(vec2 p)
{
	vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz+33.33);
    return fract((p3.xxy+p3.yzz)*p3.zyx);
}

// Created by David Hoskins and licensed under MIT.
// See https://www.shadertoy.com/view/4djSRW.
// vec3->vec3 hash function
vec3 hash33(vec3 p3)
{
	p3 = fract(p3 * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz+33.33);
    return fract((p3.xxy + p3.yxx)*p3.zyx);

}

// Created by David Hoskins and licensed under MIT.
// See https://www.shadertoy.com/view/4djSRW.
// float->vec4 hash function
vec4 hash41(float p)
{
	vec4 p4 = fract(vec4(p) * vec4(.1031, .1030, .0973, .1099));
    p4 += dot(p4, p4.wzxy+33.33);
    return fract((p4.xxyz+p4.yzzw)*p4.zywx);
    
}

// Created by David Hoskins and licensed under MIT.
// See https://www.shadertoy.com/view/4djSRW.
// vec2->vec4 hash function
vec4 hash42(vec2 p)
{
	vec4 p4 = fract(vec4(p.xyxy) * vec4(.1031, .1030, .0973, .1099));
    p4 += dot(p4, p4.wzxy+33.33);
    return fract((p4.xxyz+p4.yzzw)*p4.zywx);

}

// Created by David Hoskins and licensed under MIT.
// See https://www.shadertoy.com/view/4djSRW.
// vec3->vec4 hash function
vec4 hash43(vec3 p)
{
	vec4 p4 = fract(vec4(p.xyzx)  * vec4(.1031, .1030, .0973, .1099));
    p4 += dot(p4, p4.wzxy+33.33);
    return fract((p4.xxyz+p4.yzzw)*p4.zywx);
}

// Created by David Hoskins and licensed under MIT.
// See https://www.shadertoy.com/view/4djSRW.
// vec4->vec4 hash function
vec4 hash44(vec4 p4)
{
	p4 = fract(p4  * vec4(.1031, .1030, .0973, .1099));
    p4 += dot(p4, p4.wzxy+33.33);
    return fract((p4.xxyz+p4.yzzw)*p4.zywx);
}

// Blending functions

// Taken from https://www.iquilezles.org/www/articles/smin/smin.htm
// Exponential smooth min (k=32)
float blendExp(float a, float b, float k)
{
    float res = exp2(-k*a) + exp2(-k*b);
    return -log2(res)/k;
}

// Taken from https://www.iquilezles.org/www/articles/smin/smin.htm
// Power smooth min (k=8)
float blendPower(float a, float b, float k)
{
    a = pow(a, k);
    b = pow(b, k);
    return pow((a*b)/(a+b), 1./k);
}

// Taken from https://www.iquilezles.org/www/articles/smin/smin.htm
// Polynomial smooth min 1 (k=0.1)
float blendPolynomial1(float a, float b, float k)
{
    float h = clamp(.5+.5*(b-a)/k, 0., 1.);
    return mix(b, a, h) - k*h*(1.-h);
}

// Taken from https://www.iquilezles.org/www/articles/smin/smin.htm
// Polynomial smooth min 2 (k=0.1)
float blendPolynomial2(float a, float b, float k)
{
    float h = max(k-abs(a-b), 0.)/k;
    return min(a, b) - h*h*k*.25;
}

// Taken from https://www.iquilezles.org/www/articles/smin/smin.htm
// Generalized polynomial blending function
float blendPolynomial(float a, float b, float k, float n)
{
    float h = max(k-abs(a-b), 0.)/k;
    float m = pow(h, n)*.5;
    float s = m*k/n; 
    return (a<b)?a-s:b-s;
}

// Smooth 2D voronoi
vec3 dsmoothvoronoi2(vec2 x, float k)
{
    float n,
        ret = 1.,
        df = 10.,
        d;
    vec2 y = floor(x),
        pf=c.yy, 
        p;
    
    for(int i=-1; i<=1; i+=1)
        for(int j=-1; j<=1; j+=1)
        {
            p = y + vec2(float(i), float(j));
            p += hash12(p);
            
            d = length(x-p);
            
            if(d < df)
            {
                df = d;
                pf = p;
            }
        }
    for(int i=-1; i<=1; i+=1)
        for(int j=-1; j<=1; j+=1)
        {
            p = y + vec2(float(i), float(j));
            p += hash12(p);
            
            vec2 o = p - pf;
            d = length(.5*o-dot(x-pf, o)/dot(o,o)*o);
            ret = blendPolynomial1(ret, d, k);
        }
        
    return vec3(ret, pf);
}

// 2D voronoi
vec3 dvoronoi2(vec2 x)
{
    float n,
        ret = 1.,
        df = 10.,
        d;
    vec2 y = floor(x),
        pf=c.yy, 
        p;
    
    for(int i=-1; i<=1; i+=1)
        for(int j=-1; j<=1; j+=1)
        {
            p = y + vec2(float(i), float(j));
            p += hash12(p);
            
            d = length(x-p);
            
            if(d < df)
            {
                df = d;
                pf = p;
            }
        }
    for(int i=-1; i<=1; i+=1)
        for(int j=-1; j<=1; j+=1)
        {
            p = y + vec2(float(i), float(j));
            p += hash12(p);
            
            vec2 o = p - pf;
            d = length(.5*o-dot(x-pf, o)/dot(o,o)*o);
            ret = min(ret, d);
        }
        
    return vec3(ret, pf);
}

// 3D voronoi
vec4 dvoronoi3(vec3 x)
{
    float n,
        ret = 1.,
        df = 10.,
        d;
    vec3 y = floor(x),
        pf=c.yyy, 
        p;
    
    for(int i=-1; i<=1; i+=1)
        for(int j=-1; j<=1; j+=1)
            for(int k=-1; k<=1; k+=1)
            {
                p = y + vec3(float(i), float(j), float(k));
                p += hash13(p);

                d = length(x-p);

                if(d < df)
                {
                    df = d;
                    pf = p;
                }
            }
    for(int i=-1; i<=1; i+=1)
        for(int j=-1; j<=1; j+=1)
            for(int k=-1; k<=1; k+=1)
            {
                p = y + vec3(float(i), float(j), float(k));
                p += hash13(p);

                vec3 o = p - pf;
                d = length(.5*o-dot(x-pf, o)/dot(o,o)*o);
                ret = min(ret, d);
            }
        
    return vec4(ret, pf);
}

// Low-Frequency noise (value-type)
float lfnoise(vec2 t)
{
    vec2 i = floor(t);
    t = fract(t);
    t = smoothstep(c.yy, c.xx, t);
    vec2 v1 = vec2(hash12(i), hash12(i+c.xy)), 
        v2 = vec2(hash12(i+c.yx), hash12(i+c.xx));
    v1 = c.zz+2.*mix(v1, v2, t.y);
    return mix(v1.x, v1.y, t.x);
}

// Multi-frequency fractal noise stack
float mfnoise(vec2 x, float d, float b, float e)
{
    float n = 0.;
    float a = 1., nf = 0., buf;
    for(float f = d; f<b; f *= 2.)
    {
        n += a*lfnoise(f*x);
        a *= e;
        nf += 1.;
    }
    return n * (1.-e)/(1.-pow(e, nf));
}

// Convert RGB to HSV colors
vec3 rgb2hsv(vec3 cc)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(cc.bg, K.wz), vec4(cc.gb, K.xy), step(cc.b, cc.g));
    vec4 q = mix(vec4(p.xyw, cc.r), vec4(cc.r, p.yzx), step(p.x, cc.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

// Convert HSV to RGB colors
vec3 hsv2rgb(vec3 cc)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(cc.xxx + K.xyz) * 6.0 - K.www);
    return cc.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), cc.y);
}

// Compute an orthonormal system from a single vector in R^3
mat3 ortho(vec3 d)
{
    vec3 a = normalize(
        d.x != 0. 
            ? vec3(-d.y/d.x,1.,0.)
            : d.y != 0.
                ? vec3(1.,-d.x/d.y,0.)
                : vec3(1.,0.,-d.x/d.z)
    );
    return mat3(d, a, cross(d,a));
}

// Rotation in R3
mat3 rot3(in vec3 p)
{
    return mat3(c.xyyy, cos(p.x), sin(p.x), 0., -sin(p.x), cos(p.x))
        *mat3(cos(p.y), 0., -sin(p.y), c.yxy, sin(p.y), 0., cos(p.y))
        *mat3(cos(p.z), -sin(p.z), 0., sin(p.z), cos(p.z), c.yyyx);
}

// Originally from https://www.shadertoy.com/view/lllXz4
// Modified by fizzer to put out the vector q.
// Modified by NR4 to reduce size.
// Inverse spherical fibonacci mapping tech by las/mercury
vec2 inverseSF( vec3 p, float n, out vec3 outq ) 
{
    float m = 1. - 1./n,
        phi = min(atan(p.y, p.x), pi), cosTheta = p.z,
        k  = max(2., floor( log(n * pi * sqrt(5.) * (1.0 - cosTheta*cosTheta))/ log(PHI+1.))),
        Fk = pow(PHI, k)/sqrt(5.0),
        d,j;
    vec2  F  = vec2( round(Fk), round(Fk * PHI) ),
        ka = 2.*F/n,
        kb = 2.*pi*( fract((F+1.0)*PHI) - (PHI-1.) ),
        c;    
    mat2 iB = mat2( ka.y, -ka.x, kb.y, -kb.x ) / (ka.y*kb.x - ka.x*kb.y);
    
    c = floor( iB * vec2(phi, cosTheta - m));
    d = 8.;
    j = 0.;
    for( int s=0; s<4; s++ ) 
    {
        vec2 uv = vec2( float(s-2*(s/2)), float(s/2) );
        
        float i = round(dot(F, uv + c)),
            phi = 2.0*pi*fract(i*PHI),
            cosTheta = m - 2.0*i/n,
            sinTheta = sqrt(1.0 - cosTheta*cosTheta);
        vec3 q = vec3( cos(phi)*sinTheta, sin(phi)*sinTheta, cosTheta );
        float squaredDistance = dot(q-p, q-p);
        
        if (squaredDistance < d) 
        {
            outq = q;
            d = squaredDistance;
            j = i;
        }
    }
    return vec2( j, sqrt(d) );
}

// Analytical intersections

// Use this by plugging o-x0 into x.
// Analytical sphere distance.
vec2 asphere(vec3 x, vec3 dir, float R)
{
    vec2 dd = quadratic_zeros(vec3(dot(dir,dir),2.*dot(x,dir),dot(x,x)-R*R));
    return vec2(min(dd.x, dd.y), max(dd.x, dd.y));
}

// Use this by plugging o-x0 into x.
// Analytical box distance.
vec2 abox3(vec3 x, vec3 dir, vec3 s)
{
    vec3 a = (s-x)/dir, 
        b = -(s+x)/dir,
        dn = min(a,b),
        df = max(a,b);
    return vec2(
        all(lessThan(abs(x + dn.y * dir).zx,s.zx)) 
            ? dn.y 
            : all(lessThan(abs(x + dn.x * dir).yz,s.yz)) 
                ? dn.x 
                : all(lessThan(abs(x + dn.z * dir).xy,s.xy)) 
                    ? dn.z
                    : -1.,
        all(lessThan(abs(x + df.y * dir).zx,s.zx)) 
            ? df.y 
            : all(lessThan(abs(x + df.x * dir).yz,s.yz)) 
                ? df.x 
                : all(lessThan(abs(x + df.z * dir).xy,s.xy)) 
                    ? df.z 
                    : -1.
    );
}

// Distance functions

// 2D box distance
float dbox2(vec2 x, vec2 b)
{
    vec2 da = abs(x)-b;
    return length(max(da,c.yy)) + min(max(da.x,da.y),0.0);
}

// 3D box distance
float dbox3(vec3 x, vec3 b)
{
	vec3 da = abs(x) - b;
	return length(max(da,0.0))
			+ min(max(da.x,max(da.y,da.z)),0.0);
}

// Distance to circle segment
float dcirclesegment2(vec2 x, float r, float p0, float p1)
{
    float p = atan(x.y, x.x),
        t = 2.*pi;
    
    vec2 philo = vec2(p0, p1);
    philo = sign(philo)*floor(abs(philo)/t)*t;
    philo = vec2(min(philo.x, philo.y), max(philo.x,philo.y));
    philo.y = mix(philo.y,philo.x,.5+.5*sign(p0-p1));
    
    p0 -= philo.y;
    p1 -= philo.y;
    
    philo = vec2(max(p0, p1), min(p0, p1));
    
    if((p < philo.x && p > philo.y) 
       || (p+t < philo.x && p+t > philo.y) 
       || (p-t < philo.x && p-t > philo.y)
      )
    	return abs(length(x)-r);
    return min(
        length(x-r*vec2(cos(p0), sin(p0))),
        length(x-r*vec2(cos(p1), sin(p1)))
        );
}

// Distance to spiral
float dspiral(in vec2 x, in float k)
{
    float tau = 2.*pi;
    vec2 dpr = mod(vec2(atan(x.y,x.x),length(x)/k),tau);
    float a = abs(dpr.y-dpr.x);
    return k*min(a,tau-a);
}

// Based on the distance included in https://www.shadertoy.com/view/Xd2GR3,
// Heavily adapted for size-coding.
// Distance to hexagon pattern 
vec3 dhexagonpattern(vec2 p) 
{
    vec2 q = vec2( p.x*1.2, p.y+p.x*.6 ),
        pi = floor(q),
        pf = fract(q),
        ma,
		ind;

    float v = mod(pi.x+pi.y, 3.),
        ca = step(1.,v),
        cb = step(2.,v);

    ma = step(pf.xy,pf.yx);
    ind = (pi+ca-cb*ma)*vec2(1./1.2,1.);
	return vec3(dot(ma, 1.-pf.yx+ca*(pf.x+pf.y-1.)+cb*(pf.yx-2.*pf.xy)), vec2(ind.x, ind.y-ind.x*.6));
}

// 2D line nearest parameter
float tline2(vec2 x, vec2 p1, vec2 p2)
{
    vec2 da = p2-p1;
    return clamp(dot(x-p1, da)/dot(da,da),0.,1.);
}

// 2D line distance
float dline2(vec2 x, vec2 p1, vec2 p2)
{
    return length(x-mix(p1, p2, tline2(x,p1,p2)));
}

// 3D line nearest parameter
float tline3(vec3 x, vec3 p1, vec3 p2)
{
    vec3 da = p2-p1;
    return clamp(dot(x-p1, da)/dot(da,da),0.,1.);
}

// 3D line distance
float dline3(vec3 x, vec3 p1, vec3 p2)
{
    return length(x-mix(p1, p2, tline3(x,p1,p2)));
}

// Regular polygon distance
float dregularpolygon(in vec2 x, in float R, in float N)
{
    float p = atan(x.y,x.x),
        k = pi/N,
    	dp = mod(p+pi, 2.*k);
    
    vec2 p1 = R*c.xy,
        p2 = R*vec2(cos(2.*k),sin(2.*k)),
        dpp = p2-p1,
        n = normalize(p2-p1).yx*c.xz, 
        xp = length(x)*vec2(cos(dp), sin(dp));
    float t = dot(xp-p1,dpp)/dot(dpp,dpp);
    float r = dot(xp-p1,n);
    if(t < 0.)
        return sign(r)*length(xp-p1);
    else if(t > 1.)
        return sign(r)*length(xp-p2);
    else
	    return r;
}

// Regular star distance
float dstar(in vec2 x, in float r1, in float r2, in float N)
{
    N *= 2.;
    float p = atan(x.y,x.x),
        k = pi/N,
    	dp = mod(p+pi, 2.*k),
    	parity = mod(round((p+pi-dp)*.5/k), 2.),
        dk = k,
        dkp = mix(dk,-dk,parity);
    
    vec2 p1 = r1*vec2(cos(k-dkp),sin(k-dkp)),
        p2 = r2*vec2(cos(k+dkp),sin(k+dkp)),
        dpp = p2-p1,
        n = normalize(p2-p1).yx*c.xz, 
        xp = length(x)*vec2(cos(dp), sin(dp));
    float t = dot(xp-p1,dpp)/dot(dpp,dpp);
    float r = mix(1.,-1.,parity)*dot(xp-p1,n);
    if(t < 0.)
        return sign(r)*length(xp-p1);
    else if(t > 1.)
        return sign(r)*length(xp-p2);
    else
	    return r;
}

// 3D Point on a spline
vec3 xspline3(vec3 x, float t, vec3 p0, vec3 p1, vec3 p2)
{
    return mix(mix(p0,p1,t),mix(p1,p2,t),t);
}

// 3D Distance to a point on a spline
float dspline3(vec3 x, float t, vec3 p0, vec3 p1, vec3 p2)
{
    return length(x - xspline3(x, t, p0, p1, p2));
}

// 3D Normal in a point on a spline
vec3 nspline3(vec3 x, float t, vec3 p0, vec3 p1, vec3 p2)
{
    return normalize(mix(p1-p0, p2-p1, t));
}

// Returns vec2(dmin, tmin).
// 3D spline parameter of the point with minimum distance on the spline and sdf
vec2 dtspline3(vec3 x, vec3 p0, vec3 p1, vec3 p2)
{
    vec3 E = x-p0, F = p2-2.*p1+p0, G = p1-p0;
    E = clamp(cubic_zeros(vec4(dot(F,F), 3.*dot(G,F), 2.*dot(G,G)-dot(E,F), -dot(E,G))),0.,1.);
    F = vec3(dspline3(x,E.x,p0,p1,p2),dspline3(x,E.y,p0,p1,p2),dspline3(x,E.z,p0,p1,p2));
    return F.x < F.y && F.x < F.z
        ? vec2(F.x, E.x)
        : F.y < F.x && F.y < F.z
            ? vec2(F.y, E.y)
            : vec2(F.z, E.z);
}

// 2D Point on a spline
vec2 xspline2(vec2 x, float t, vec2 p0, vec2 p1, vec2 p2)
{
    return mix(mix(p0,p1,t),mix(p1,p2,t),t);
}

// 2D Distance to a point on a spline
float dspline2(vec2 x, float t, vec2 p0, vec2 p1, vec2 p2)
{
    return length(x - xspline2(x, t, p0, p1, p2));
}

// 2D Normal in a point on a spline
vec2 nspline2(vec2 x, float t, vec2 p0, vec2 p1, vec2 p2)
{
    return normalize(mix(p1-p0, p2-p1, t));
}

// Returns vec2(dmin, tmin).
// 2D spline parameter of the point with minimum distance on the spline and sdf
vec2 dtspline2(vec2 x, vec2 p0, vec2 p1, vec2 p2)
{
    vec2 E0 = x-p0, F0 = p2-2.*p1+p0, G0 = p1-p0;
    vec3 E = clamp(cubic_zeros(vec4(dot(F0,F0), 3.*dot(G0,F0), 2.*dot(G0,G0)-dot(E0,F0), -dot(E0,G0))),0.,1.),
        F = vec3(dspline2(x,E.x,p0,p1,p2),dspline2(x,E.y,p0,p1,p2),dspline2(x,E.z,p0,p1,p2));
    return F.x < F.y && F.x < F.z
        ? vec2(F.x, E.x)
        : F.y < F.x && F.y < F.z
            ? vec2(F.y, E.y)
            : vec2(F.z, E.z);
}

// Operators for CSG

// Unite two sdfs.
vec2 add2(vec2 sda, vec2 sdb)
{
    return (sda.x<sdb.x)?sda:sdb;
}

// Unite two sdfs.
vec3 add3(vec3 sda, vec3 sdb)
{
	return (sda.x<sdb.x)?sda:sdb;
}

// Unite two sdfs.
vec4 add4(vec4 sda, vec4 sdb)
{
	return (sda.x<sdb.x)?sda:sdb;
}

// Cut two sdfs.
vec2 sub2(vec2 sda, vec2 sdb)
{
    return (sda.x>sdb.x)?abs(sda):abs(sdb)*c.zx;
}

// Cut two sdfs.
vec3 sub3(vec3 sda, vec3 sdb)
{
    return (sda.x>sdb.x)?abs(sda):abs(sdb)*c.zxx;
}

// Cut two sdfs.
vec4 sub4(vec4 sda, vec4 sdb)
{
    return (sda.x>sdb.x)?abs(sda):abs(sdb)*c.zxxx;
}

// Extrude sdf along axis
float zextrude(float z, float d2d, float h)
{
    vec2 w = vec2(d2d, abs(z)-0.5*h);
    return min(max(w.x,w.y),0.0) + length(max(w,0.0));
}

// Paint with antialiasing
float sm(in float d)
{
    return smoothstep(1.5/iResolution.y, -1.5/iResolution.y, d);
}
