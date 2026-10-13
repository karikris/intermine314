"""Sized Expat parsing with at most one selected subtree in memory."""
from collections import deque
from xml.etree.ElementTree import TreeBuilder
from xml.parsers.expat import ParserCreate


def xml_events(stream):
    """Yield XML events from bounded reads, retaining Expat error semantics."""
    events = deque()
    parser = ParserCreate()
    parser.StartElementHandler = lambda name, attrs: events.append(('start', name, attrs))
    parser.EndElementHandler = lambda name: events.append(('end', name, None))
    parser.CharacterDataHandler = lambda text: events.append(('text', text, None))
    parser.CommentHandler = lambda text: events.append(('comment', text, None))
    parser.ProcessingInstructionHandler = lambda target, text: events.append(('pi', target, text))
    while True:
        chunk = stream.read(65536)
        parser.Parse(chunk, not chunk)
        while events:
            yield events.popleft()
        if not chunk:
            return


def selected_elements(events, tag):
    """Build only an outer selected subtree, then yield nested matches in order."""
    builder = None
    depth = 0
    for event, name, value in events:
        if event == 'start':
            if builder is None and name == tag:
                builder = TreeBuilder(insert_comments=True, insert_pis=True)
            if builder is not None:
                builder.start(name, value)
                depth += 1
        elif builder is not None:
            if event == 'end':
                builder.end(name)
                depth -= 1
                if not depth:
                    element = builder.close()
                    builder = None
                    yield from element.iter(tag)
            elif event == 'text':
                builder.data(name)
            elif event == 'comment':
                builder.comment(name)
            elif event == 'pi':
                builder.pi(name, value)
