# ======================================================================
#         Imports
# ======================================================================
import numpy as np
from .baseConstraint import GeometricConstraint
from pygeo.geo_utils import convertTo1D


class AreaMomentsConstraint(GeometricConstraint):
    """
    This class is used to represent individual second moment of area constraint.
    The parameter list is explained in the addAreaMomentConstraint() of
    the DVConstraints class
    """

    def __init__(self, name, nSpan, nChord, coords, lower, upper, scaled, scale, DVGeo, addToPyOpt):
        super().__init__(name, nSpan, lower, upper, scale, DVGeo, addToPyOpt)

        self.nSpan = nSpan
        self.nChord = nChord
        self.coords = coords
        self.scaled = scaled

        self.supportedKeys = ["Ixx", "Iyy", "Jz"]

        # Before doing anything check and convert the bounds to a dict format
        self.lower = self._checkAndConvertBound(self.lower)
        self.upper = self._checkAndConvertBound(self.upper)

        # First thing we can do is embed the coordinates into DVGeo
        # with the name provided
        self.DVGeo.addPointSet(self.coords, self.name)

        # Now compute the reference second moments of area
        self.Ixx0, self.Iyy0, self.Jz0 = self.evalAreaMoments()

        # IJ should have three components: Ixx, Iyy, and J

    def evalFunctions(self, funcs, config):
        """
        Evaluate the function this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates
        self.coords = self.DVGeo.update(self.name, config=config)

        # Evaluate and scale if needed before returning
        Ixx, Iyy, Jz = self.evalAreaMoments()
        if self.scaled:
            Ixx = np.divide(Ixx, self.Ixx0)
            Iyy = np.divide(Iyy, self.Iyy0)
            Jz = np.divide(Jz, self.Jz0)

        funcs[f"{self.name}_Ixx"] = Ixx
        funcs[f"{self.name}_Iyy"] = Iyy
        funcs[f"{self.name}_Jz"] = Jz

    def evalAreaMoments(self):
        """
        Evaluate the second moments of area for each section at all spanwise locations
        """
        # IJ = {} # TODO: could make these three into a dictionary of arrays with Ixx, etc label?
        Ixx = np.zeros(self.nSpan)
        Iyy = np.zeros(self.nSpan)
        Jz = np.zeros(self.nSpan)

        # Reshape coordinates for convenience
        coords = self.coords.reshape((self.nSpan, self.nChord, 2, 3))

        # Loop over each spanwise location
        for i in range(self.nSpan):

            # [] TODO JM-: For RAE2822 airfoil, I had to flip the top/bottom index (3rd index below) to get positive dy. Need to test again in box and other cases. Is there a consistent definition?
            # EJ-JM: This depends on how the coordinates are written. We may need to consider different order ijk

            # Chordwise coordinate
            x_bot = coords[i, :, 1, 0]
            x_top = coords[i, :, 0, 0]
            # Vertical coordinate
            y_bot = coords[i, :, 1, 1]
            y_top = coords[i, :, 0, 1]

            # Evaluate
            Ixx[i], Iyy[i], Jz[i] = self._evalAreaMomentsSection(x_top, x_bot, y_top, y_bot)

        return Ixx, Iyy, Jz

    def _evalAreaMomentsSection(self, x_top, x_bot, y_top, y_bot):
        """
        This function computes the area moments of a section (wing or airfoil section)
        approximated by one or more rectangles.

              /                        /
        Ixx = | y^2 dx dy    ,   Iyy = | x^2 dx dy    , J = <from book of formulae based on shape>
              /                        /
        """
        # EJ-: This function should be generalized, such that different approximations can be used. Also, should refactor to include geometric objects.
        # GN: What is meant by geometric objects? Also, this only works for solid sections, right?

        # Initialize output
        Ixx = 0
        Iyy = 0
        Jz = 0

        # Section centroid
        xc = 0
        yc = 0
        # Section area and chord
        area = 0
        chord = 0

        # Loop over chordwise indices for current sections
        for j in range(self.nChord - 1):

            # Rectangular strip with midpoint values
            # Sides
            dx = x_bot[j + 1] - x_bot[j]
            chord += dx

            # Sanity checks for a rectangular strip
            assert dx > 0, "dx is not positive!"
            np.testing.assert_allclose(
                dx, x_top[j + 1] - x_top[j], err_msg=f"dx not same on top and bottom for chord:{j}!"
            )

            # y coordinate in middle top of rectangle
            y_rect_top = (y_top[j] + y_top[j + 1]) / 2
            # y coordinate in middle bottom of rectangle
            y_rect_bot = (y_bot[j] + y_bot[j + 1]) / 2
            # Height of rectangle
            dy_rect = y_rect_top - y_rect_bot
            assert dy_rect > 0, "dy_rect is not positive!"

            # Area of rectangular strip
            area_rect = dx * dy_rect
            area += area_rect

            # Centroid of rectangular strip
            xc_rect = x_bot[j] + dx / 2
            yc_rect = y_rect_bot + dy_rect / 2

            # Add centroid of rectangular strip
            xc += xc_rect * area_rect
            yc += yc_rect * area_rect

            # Second moments of area about global coordinates via parallel axis theorem (e.g. I_glob = I_cent + [A] [y_c^2])
            Ixx += (dx * dy_rect**3) / 12 + area_rect * yc_rect**2
            Iyy += (dx**3 * dy_rect) / 12 + area_rect * xc_rect**2

            # Drela's version of St. Venant's torsional constant TODO: provide source?
            # This link says this formula is the torsion constant for a rectangle of very low thickness-chord ratio
            # https://en.wikipedia.org/wiki/Torsion_constant#cite_note-7:~:text=the%20side%20length.-,Rectangle,-%5Bedit%5D
            Jz += dy_rect**3 * dx / 3

        # Compute centroid
        xc /= area
        yc /= area
        # Convert moments of area to be relative to airfoil centroid
        Ixx -= area * yc**2
        Iyy -= area * xc**2

        # Approximate torsional rigidity based on one rectangular strip
        # t_avg = area / chord
        # Jz[i] = chord * t_avg**3 / 3

        return Ixx, Iyy, Jz

    def evalFunctionsSens(self, funcsSens, config):
        pass

        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """
        nDV = self.DVGeo.getNDV()
        if nDV > 0:

            # Compute all Jacobians
            dIxxdPt = self.evalAreaMomentsSens(Ixxb=1.0, Iyyb=0.0, Jzb=0.0)
            dIyydPt = self.evalAreaMomentsSens(Ixxb=0.0, Iyyb=1.0, Jzb=0.0)
            dJzdPt = self.evalAreaMomentsSens(Ixxb=0.0, Iyyb=0.0, Jzb=1.0)

            if self.scaled:
                for i in range(self.nSpan):
                    dIxxdPt[i, :, :] /= self.Ixx0[i]
                    dIyydPt[i, :, :] /= self.Iyy0[i]
                    dJzdPt[i, :, :] /= self.Jz0[i]

            # Now compute the DVGeo total sensitivity
            funcsSens[f"{self.name}_Ixx"] = self.DVGeo.totalSensitivity(dIxxdPt, self.name, config=config)
            funcsSens[f"{self.name}_Iyy"] = self.DVGeo.totalSensitivity(dIyydPt, self.name, config=config)
            funcsSens[f"{self.name}_Jz"] = self.DVGeo.totalSensitivity(dJzdPt, self.name, config=config)

    def evalAreaMomentsSens(self, Ixxb=0.0, Iyyb=0.0, Jzb=0.0):
        """
        Evaluate the derivative of the second moments of area with respect to the
        coordinates given a seed.
        """

        # Reshape coordinates
        coords = self.coords.reshape((self.nSpan, self.nChord, 2, 3))

        # Allocate and do all span locations at once
        coordsb = np.zeros((self.nSpan, self.coords.shape[0], self.coords.shape[1]))

        # Loop over each spanwise location
        for i in range(self.nSpan):

            tempb = np.zeros_like(coords)

            # x_top is not needed as its not used in anything. Its only function is to make sure x_top and x_bot are the same in
            # x-value. This is why its not present here for derivatives.

            # Chordwise coordinate
            x_bot = coords[i, :, 1, 0]
            # Vertical coordinate
            y_bot = coords[i, :, 1, 1]
            y_top = coords[i, :, 0, 1]

            x_botb = np.zeros_like(x_bot)
            y_botb = np.zeros_like(y_bot)
            y_topb = np.zeros_like(y_top)

            # GN: I don't know what's going on here
            self._evalAreaMomentsSection_b(x_bot, x_botb, y_top, y_topb, y_bot, y_botb, Ixxb, Iyyb, Jzb)

            # Set the derivatives in the global coordinate array for this slice
            tempb[i, :, 1, 0] = x_botb
            tempb[i, :, 1, 1] = y_botb
            tempb[i, :, 0, 1] = y_topb

            # Reshape back to flattened array for DVGeo
            coordsb[i, :, :] = tempb.reshape((self.nSpan * self.nChord * 2, 3))

        return coordsb

    def _evalAreaMomentsSection_b(self, x_bot, x_botb, y_top, y_topb, y_bot, y_botb, Ixxb, Iyyb, Jzb):

        # Initialize stack for storing intermediate variables
        stack = []

        # Section area and chord
        area = 0.0
        # Section centroid
        xc = 0.0
        yc = 0.0

        # Loop over chordwise indices for current section
        for j in range(self.nChord - 1):
            # Rectangular strip with midpoint values
            # Sides
            dx = x_bot[j + 1] - x_bot[j]
            # y coordinate in middle top of rectangle
            y_rect_top = (y_top[j] + y_top[j + 1]) / 2
            # y coordinate in middle bottom of rectangle
            y_rect_bot = (y_bot[j] + y_bot[j + 1]) / 2
            # Height of rectangle
            dy_rect = y_rect_top - y_rect_bot
            # Area of rectangular strip
            area_rect = dx * dy_rect
            area = area + area_rect
            # Centroid of rectangular strip
            xc_rect = x_bot[j] + dx / 2
            yc_rect = y_rect_bot + dy_rect / 2
            # Add centroid of rectangular strip
            xc = xc + xc_rect * area_rect
            yc = yc + yc_rect * area_rect

        stack.append(xc)
        xc = xc / area
        stack.append(yc)
        yc = yc / area

        # GN: More comments here as to what is going on?
        areab = -(xc**2 * Iyyb) - yc**2 * Ixxb
        xcb = -(2 * xc * area * Iyyb)
        ycb = -(2 * yc * area * Ixxb)
        yc = stack.pop()
        xc = stack.pop()
        areab = areab - yc * ycb / area**2 - xc * xcb / area**2
        ycb = ycb / area
        xcb = xcb / area
        for j in range(self.nChord - 2, -1, -1):
            dx = x_bot[j + 1] - x_bot[j]
            y_rect_top = (y_top[j] + y_top[j + 1]) / 2
            y_rect_bot = (y_bot[j] + y_bot[j + 1]) / 2
            dy_rect = y_rect_top - y_rect_bot
            area_rect = dx * dy_rect
            xc_rect = x_bot[j] + dx / 2
            xc_rectb = 2 * xc_rect * area_rect * Iyyb + area_rect * xcb
            yc_rect = y_rect_bot + dy_rect / 2
            area_rectb = xc_rect**2 * Iyyb + yc_rect**2 * Ixxb + yc_rect * ycb + xc_rect * xcb + areab
            dxb = (
                dy_rect**3 * Jzb / 3
                + 3 * dx**2 * dy_rect * Iyyb / 12
                + dy_rect**3 * Ixxb / 12
                + xc_rectb / 2
                + dy_rect * area_rectb
            )
            yc_rectb = 2 * yc_rect * area_rect * Ixxb + area_rect * ycb
            dy_rectb = (
                dy_rect**2 * dx * Jzb
                + dx**3 * Iyyb / 12
                + 3 * dy_rect**2 * dx * Ixxb / 12
                + yc_rectb / 2
                + dx * area_rectb
            )
            y_rect_botb = yc_rectb - dy_rectb
            x_botb[j] = x_botb[j] + xc_rectb - dxb
            y_rect_topb = dy_rectb
            y_botb[j] = y_botb[j] + y_rect_botb / 2
            y_botb[j + 1] = y_botb[j + 1] + y_rect_botb / 2
            y_topb[j] = y_topb[j] + y_rect_topb / 2
            y_topb[j + 1] = y_topb[j + 1] + y_rect_topb / 2
            x_botb[j + 1] = x_botb[j + 1] + dxb

        Jzb = 0.0
        Ixxb = 0.0
        Iyyb = 0.0

    def _checkAndConvertBound(self, bound):
        """Check the bound has the correct dict format and length"""

        # Initialize output
        convertedBound = {}

        # If bound is a single float value (as the default) we make sure to
        # populate all constraints with this bound and return
        if isinstance(bound, float) or isinstance(bound, int):
            tmp = bound * np.ones(self.nSpan)
            for key in self.supportedKeys:
                convertedBound[key] = tmp
            return convertedBound

        # If not a float we need to have a dict
        if not isinstance(bound, dict):
            raise TypeError(f"The supplied bound should be of type dict, got {type(bound)}")

        # Bound is a dict, check that given keys are supported
        for key in self.supportedKeys:
            # If a key is not present we assume that there is no bound and set None
            if key not in bound:
                convertedBound[key] = None
            else:
                # We either support a list or a float of values
                convertedBound[key] = convertTo1D(bound[key], self.nSpan)

        return convertedBound

    def addConstraintsPyOpt(self, optProb, exclude_wrt=None):
        """
        Add the constraints to pyOpt, if the flag is set
        """
        if self.addToPyOpt:
            wrt = self.getVarNames(exclude_wrt)

            for key in self.supportedKeys:
                optProb.addConGroup(
                    self.name + f"_{key}",
                    self.nCon,
                    lower=self.lower[key],
                    upper=self.upper[key],
                    scale=self.scale,
                    wrt=wrt,
                )

    def writeTecplot(self, handle):
        """
        Writes the input coordinates that are used to compute the area moment values for slices.
        The representation is similar to a volume, but it does not capture the slices properly. 
        Needs to be refactored to show the slices/rectangles used.
        """
        # GN: Should actually shade in the slice/rectangles used instead of the bounding box IMO

        x = self.coords.reshape([self.nSpan, self.nChord, 2, 3])

        handle.write(f'ZONE T="{self.name}" I={self.nSpan} J={self.nChord} K=2\n')
        handle.write("DATAPACKING=POINT\n")
        for k in range(2):
            for j in range(self.nChord):
                for i in range(self.nSpan):
                    handle.write(f"{x[i, j, k, 0]:f} {x[i, j, k, 1]:f} {x[i, j, k, 2]:f}\n")
