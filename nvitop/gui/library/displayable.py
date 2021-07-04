# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# This file is originally part of ranger, the console file manager. https://github.com/ranger/ranger
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-function-docstring
# pylint: disable=invalid-name

from nvitop.gui.library.libcurses import CursesShortcuts


class Displayable(CursesShortcuts):
    """Displayables are objects which are displayed on the screen.

    This is just the abstract class, defining basic operations
    such as resizing, printing, changing colors.
    Subclasses of displayable can extend these methods:

    draw() -- draw the object. Is only called if visible.
    poke() -- is called just before draw(), even if not visible.
    finalize() -- called after all objects finished drawing.
    press(key) -- called after a key press on focused objects.
    destroy() -- called before destroying the displayable object

    Additionally, there are these methods:

    __contains__(item) -- is the item (y, x) inside the panel?

    These attributes are set:

    Modifiable:
        focused -- Focused objects receive press() calls.
        visible -- Visible objects receive draw() and finalize() calls
        need_redraw -- Should the panel be redrawn? This variable may
            be set at various places in the script and should eventually be
            handled (and unset) in the draw() method.

    Read-Only: (i.e. recommended not to change manually)
        win -- the own curses window object
        parent -- the parent (DisplayableContainer) object or None
        x, y, width, height -- absolute coordinates and boundaries
    """

    def __init__(self, win, root=None):
        super().__init__()

        self._need_redraw = True
        self.focused = False
        self._old_visible = self._visible = True
        self.x = 0
        self.y = 0
        self._width = 0
        self.height = 0

        self.win = win
        self.root = root
        self.parent = None

    def __contains__(self, item):
        """Checks if item is inside the boundaries.

        item can be an iterable like [y, x] or an object with x and y methods.
        """
        try:
            y, x = item.y, item.x
        except AttributeError:
            try:
                y, x = item
            except (ValueError, TypeError):
                return False

        return self.contains_point(y, x)

    def contains_point(self, y, x):
        """Test whether the point lies inside this object.

        x and y should be absolute coordinates.
        """
        return (self.x <= x < self.x + self.width) and (self.y <= y < self.y + self.height)

    def poke(self):
        """Called before drawing, even if invisible"""

        if self._old_visible != self.visible:
            self._old_visible = self.visible
            self.need_redraw = True

            if not self.visible:
                self.win.erase()

    def draw(self):
        """Draw the object.

        Called on every main iteration if visible. Containers should call draw()
        on their contained objects here. Override this!
        """

        self.need_redraw = False

    def finalize(self):
        """Called after every displayable is done drawing.

        Override this!
        """

        self.need_redraw = False

    def destroy(self):
        """Called when the object is destroyed."""

        self.win = None
        self.root = None

    def click(self, event):
        """Called when a mouse key is pressed and self.focused is True.

        Override this!
        """

    def press(self, key):
        """Called when a key is pressed and self.focused is True.

        Override this!
        """

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value):
        if self._visible != value:
            self.need_redraw = True
            self._visible = value
        if not self.visible:
            self.focused = False

    @property
    def need_redraw(self):
        return self._need_redraw

    @need_redraw.setter
    def need_redraw(self, value):
        if self._need_redraw != value:
            self._need_redraw = value
            if value and self.parent is not None and not self.parent.need_redraw:
                self.parent.need_redraw = True

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        if self.width != value and self.visible:
            self.need_redraw = True
        self._width = value

    def __str__(self):
        return self.__class__.__name__


class DisplayableContainer(Displayable):
    """DisplayableContainers are Displayables which contain other Displayables.

    This is also an abstract class. The methods draw, poke, finalize,
    click, press and destroy are extended here and will recursively
    call the function on all contained objects.

    New methods:

    add_child(object) -- add the object to the container.
    replace_child(old_obj, new_obj) -- replaces old object with new object.
    remove_child(object) -- remove the object from the container.

    New attributes:

    container -- a list with all contained objects (rw)
    """

    def __init__(self, win, root=None):
        super().__init__(win, root)

        self.container = []

    # extended or overridden methods

    def poke(self):
        """Recursively called on objects in container"""

        super().poke()
        for displayable in self.container:
            displayable.poke()

    def draw(self):
        """Recursively called on visible objects in container"""

        for displayable in self.container:
            if self.need_redraw:
                displayable.need_redraw = True
            if displayable.visible:
                displayable.draw()

        self.need_redraw = False

    def finalize(self):
        """Recursively called on visible objects in container"""

        for displayable in self.container:
            if displayable.visible:
                displayable.finalize()

    def destroy(self):
        """Recursively called on objects in container"""

        for displayable in self.container:
            displayable.destroy()
        super().destroy()

    def press(self, key):
        """Recursively called on objects in container"""

        focused_obj = self.get_focused_obj()

        if focused_obj:
            focused_obj.press(key)
            return True
        return False

    def click(self, event):
        """Recursively called on objects in container"""

        focused_obj = self.get_focused_obj()
        if focused_obj and focused_obj.click(event):
            return True

        for displayable in self.container:
            if displayable.visible and event in displayable:
                if displayable.click(event):
                    return True

        return False

    # new methods

    def add_child(self, obj):
        """Add the objects to the container."""

        if obj.parent is not None:
            obj.parent.remove_child(obj)
        self.container.append(obj)
        obj.parent = self
        obj.root = self.root

    def replace_child(self, old_obj, new_obj):
        """Replace the old object with the new instance in the container."""

        self.container[self.container.index(old_obj)] = new_obj
        new_obj.parent = self
        new_obj.root = self.root

    def remove_child(self, obj):
        """Remove the object from the container."""

        try:
            self.container.remove(obj)
        except ValueError:
            pass
        else:
            obj.parent = None
            obj.root = None

    def get_focused_obj(self):
        # Finds a focused displayable object in the container.
        for displayable in self.container:
            if displayable.focused:
                return displayable
            try:
                obj = displayable.get_focused_obj()
            except AttributeError:
                pass
            else:
                if obj is not None:
                    return obj
        return None
