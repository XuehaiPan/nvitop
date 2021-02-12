# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# This file is originally part of ranger, the console file manager. https://github.com/ranger/ranger
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name

import curses


DEFAULT_FOREGROUND = curses.COLOR_WHITE
DEFAULT_BACKGROUND = curses.COLOR_BLACK
COLOR_PAIRS = {None: 0}


def get_color(fg, bg):
    """Returns the curses color pair for the given fg/bg combination."""

    if isinstance(fg, str):
        fg = getattr(curses, 'COLOR_{}'.format(fg.upper()), -1)
    if isinstance(bg, str):
        bg = getattr(curses, 'COLOR_{}'.format(bg.upper()), -1)

    key = (fg, bg)
    if key not in COLOR_PAIRS:
        size = len(COLOR_PAIRS)
        try:
            curses.init_pair(size, fg, bg)
        except curses.error:
            # If curses.use_default_colors() failed during the initialization
            # of curses, then using -1 as fg or bg will fail as well, which
            # we need to handle with fallback-defaults:
            if fg == -1:  # -1 is the "default" color
                fg = DEFAULT_FOREGROUND
            if bg == -1:  # -1 is the "default" color
                bg = DEFAULT_BACKGROUND

            try:
                curses.init_pair(size, fg, bg)
            except curses.error:
                # If this fails too, colors are probably not supported
                pass
        COLOR_PAIRS[key] = size

    return COLOR_PAIRS[key]


def get_color_attr(fg=-1, bg=-1, attr=0):
    """Returns the curses attribute for the given fg/bg/attr combination."""
    if isinstance(attr, str):
        attr_strings = map(str.strip, attr.split('|'))
        attr = 0
        for s in attr_strings:
            attr |= getattr(curses, 'A_{}'.format(s.upper()), 0)
    if fg == -1 and bg == -1:
        return attr
    return curses.color_pair(get_color(fg, bg)) | attr


class CursesShortcuts(object):
    """This class defines shortcuts to facilitate operations with curses.

    color(*keys) -- sets the color associated with the keys from
        the current colorscheme.
    color_at(y, x, wid, *keys) -- sets the color at the given position
    color_reset() -- resets the color to the default
    addstr(*args) -- failsafe version of self.win.addstr(*args)
    """

    def __init__(self):
        self.win = None

    def addstr(self, *args, **kwargs):
        try:
            self.win.addstr(*args, **kwargs)
        except curses.error:
            pass

    def addnstr(self, *args, **kwargs):
        try:
            self.win.addnstr(*args, **kwargs)
        except curses.error:
            pass

    def addch(self, *args, **kwargs):
        try:
            self.win.addch(*args, **kwargs)
        except curses.error:
            pass

    def color(self, fg=-1, bg=-1, attr=0):
        """Change the colors from now on."""
        self.set_fg_bg_attr(fg, bg, attr)

    def color_at(self, y, x, width, *args, **kwargs):
        """Change the colors at the specified position"""
        try:
            self.win.chgat(y, x, width, get_color_attr(*args, **kwargs))
        except curses.error:
            pass

    def set_fg_bg_attr(self, fg=-1, bg=-1, attr=0):
        try:
            self.win.attrset(get_color_attr(fg, bg, attr))
        except curses.error:
            pass

    def color_reset(self):
        """Change the colors to the default colors"""
        self.color()


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
        super(Displayable, self).__init__()

        self._need_redraw = True
        self.focused = False
        self._visible = True
        self.x = 0
        self.y = 0
        self.width = 0
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
        if not self.visible and self.need_redraw:
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

    @property
    def need_redraw(self):
        return self._need_redraw

    @need_redraw.setter
    def need_redraw(self, value):
        if self._need_redraw != value:
            self._need_redraw = value
            if value and self.parent is not None:
                self.parent.need_redraw = True

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
        super(DisplayableContainer, self).__init__(win, root)

        self.container = []

    # extended or overridden methods

    def poke(self):
        """Recursively called on objects in container"""
        super(DisplayableContainer, self).poke()
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
        super(DisplayableContainer, self).destroy()

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
