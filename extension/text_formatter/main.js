define([
    'base/js/namespace',
    'jquery',
], function(Jupyter, $) {
    "use strict";

    function getActiveCell() {
        return Jupyter.notebook.get_selected_cell().code_mirror;
    }

    var toggleTag = function(tag, cm) {
        var text = cm.getSelection();
        var newText = text.replace(new RegExp(`<${tag}>`, 'g'), '').replace(new RegExp(`</${tag}>`, 'g'), '');
        cm.replaceSelection(`<${tag}>` + newText + `</${tag}>`);
    };

    var load_extension = function() {
        Jupyter.toolbar.add_buttons_group([
            {
                'label'  : '',
                'icon'   : 'fa-bold',
                'callback': function () {
                    var cm = getActiveCell();
                    toggleTag('b', cm);
                }
            },
            {
                'label'  : '',
                'icon'   : 'fa-italic',
                'callback': function () {
                    var cm = getActiveCell();
                    toggleTag('i', cm);
                }
            },
            {
                'label'  : '',
                'icon'   : 'fa-underline',
                'callback': function () {
                    var cm = getActiveCell();
                    toggleTag('u', cm);
                }
            },
            {
                'label'  : 'Quote',
                'icon'   : 'fa-quote-left',
                'callback': function () {
                    var cm = getActiveCell();
                    var text = cm.getSelection();
                    cm.replaceSelection('<blockquote>' + text + '</blockquote>');
                }
            },
            {
                'label'  : 'Code',
                'icon'   : 'fa-code',
                'callback': function () {
                    var cm = getActiveCell();
                    var text = cm.getSelection();
                    cm.replaceSelection('<pre><code>' + text + '</code></pre>');
                }
            },
            {
                'label'  : '1',
                'icon'   : 'fa-header',
                'text'   : '1',
                'callback': function () {
                    var cm = getActiveCell();
                    var text = cm.getSelection();
                    cm.replaceSelection('<h1>' + text + '</h1>');
                }
            },
            {
                'label'  : '2',
                'icon'   : 'fa-header',
                'text'   : '2',
                'callback': function () {
                    var cm = getActiveCell();
                    var text = cm.getSelection();
                    cm.replaceSelection('<h2>' + text + '</h2>');
                }
            },
            {
                'label'  : '3',
                'icon'   : 'fa-header',
                'text'   : '3',
                'callback': function () {
                    var cm = getActiveCell();
                    var text = cm.getSelection();
                    cm.replaceSelection('<h3>' + text + '</h3>');
                }
            },
            {
                'label'  : 'DarkGray',
                'icon'   : 'fa-paint-brush',
                'callback': function () {
                    var cm = getActiveCell();
                    var text = cm.getSelection();
                    cm.replaceSelection('<div style="background-color: #5A5A5A; color: white">' + text + '</div>');
                }
            },
            {
                'label'  : 'o',
                'icon'   : 'fa-font',
                'text'   : 'o',
                'callback': function () {
                    var cm = getActiveCell();
                    var text = cm.getSelection();
                    cm.replaceSelection('<span style="color: orange">' + text + '</span>');
                }
            },
            {
                'label'  : 'b',
                'icon'   : 'fa-font',
                'text'   : 'b',
                'callback': function () {
                    var cm = getActiveCell();
                    var text = cm.getSelection();
                    cm.replaceSelection('<span style="color: #1d1da3">' + text + '</span>');
                }
            }
        ]);
    };

    var extension = {
        load_ipython_extension : load_extension
    };
    return extension;
});
