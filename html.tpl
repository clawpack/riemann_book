{%- extends 'full.tpl' -%}

{%- block html_head -%}
<meta charset="utf-8" />
<title>{{resources['metadata']['name']}}</title>

{% for css in resources.inlining.css -%}
    <style type="text/css">
    {{ css }}
    </style>
{% endfor %}

<style type="text/css">
/* Overrides of notebook CSS for static HTML export */
body {
  overflow: visible;
  padding: 8px;
}
div#notebook {
  overflow: visible;
  border-top: none;
}
{%- if resources.global_content_filter.no_prompt-%}
div#notebook-container{
  padding: 6ex 12ex 8ex 12ex;
}
{%- endif -%}
@media print {
  div.cell {
    display: block;
    page-break-inside: avoid;
  }
  div.output_wrapper {
    display: block;
    page-break-inside: avoid;
  }
  div.output {
    display: block;
    page-break-inside: avoid;
  }
}
</style>

<!-- Custom stylesheet, it must be in the same directory as the html file -->
<link rel="stylesheet" href="custom.css">

    <!-- Load mathjax -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS_HTML"></script>
    <!-- MathJax configuration -->
    <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
            inlineMath: [ ['$','$'], ["\\(","\\)"] ],
            displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
            processEscapes: true,
            processEnvironments: true
        },
        // Center justify equations in code and markdown cells. Elsewhere
        // we use CSS to left justify single line equations in code cells.
        displayAlign: 'center',
        "HTML-CSS": {
            styles: {'.MathJax_Display': {"margin": 0}},
            linebreaks: { automatic: true }
        },
      TeX: { equationNumbers: { autoNumber: "AMS" } }
    });
    </script>
    <!-- End of mathjax configuration -->
{%- endblock html_head -%}

{% block body %}
{{ super() }}

<!-- Loads nbinteract package -->
<script src="https://unpkg.com/nbinteract-core"></script>
<script>
  var interact = new NbInteract({
    spec: 'Calebs97/riemann_book/master',
  })
  interact.prepare()
</script>

{%- endblock body %}

{# Add loading button to widget output #}
{%- block data_widget_view scoped %}
<div class="output_subarea output_widget_view {{ extra_class }}">
  <button class="js-nbinteract-widget">
    Show Widget
  </button>
</div>
{%- endblock data_widget_view -%}
