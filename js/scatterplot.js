// To see page: https://annemijnd.github.io/codesample/html/scat_freq_acc.html

// Define data path and import data
var datapath = "../";
try {
  d3.csv(datapath + "results/features/heatmap_InfoGain_freq.csv", function(data) {
    console.log(data)
  })
}
catch(err) {
  datapath = "../../"
  d3.csv(datapath + "results/features/heatmap_InfoGain_freq.csv", function(data) {
    console.log(data)})
}

d3.csv(datapath + 'results/features/IG100.csv', function loadCallback(error, data) {
    data.forEach(function(d) {
        // convert strings to numbers
        d.accs = +d.accs;
        d.freqs = +d.freqs;
        d.features = +d.features;
    });
    makeVis2(data);
});

// Common pattern for defining vis size and margins
var makeVis2 = function(data) {

    var margin = { top: 100, right: 20, bottom: 100, left: 100 },
        width  = 960 - margin.left - margin.right,
        height = 600 - margin.top - margin.bottom;

    // Add the visualization svg canvas to the vis-container <div>
    var canvas = d3.select("#vis-container").append("svg")
        .attr("width",  width  + margin.left + margin.right)
        .attr("height", height + margin.top  + margin.bottom)
      .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    // Define our scales
    var colorScale = d3.scale.category10();

    var xScale = d3.scale.linear()
        .domain([ 0, d3.max(data, function(d) { return d.freqs; }) + 1 ])
        .range([0, width])
        .nice();


    var yScale = d3.scale.linear()
        .domain([ 0,1 ])
        .range([height, 0])
        .nice();

    // Define our axes
    var xAxis = d3.svg.axis()
        .scale(xScale)
        .orient('bottom');

    var yAxis = d3.svg.axis()
        .scale(yScale)
        .orient('left');


    // Add x-axis to the canvas
    canvas.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis)
      .append("text")
        .attr("class", "label")
        .attr("x", width/2 + 30)
        .attr("y", 50)
        .style("text-anchor", "end")
        .text("Frequency");

    // Add y-axis to the canvas
    canvas.append("g")
        .attr("class", "y axis")
        .call(yAxis)
        .append("text")
        .attr("class", "label")
        .attr("transform", "rotate(-90)")
        .attr("y",  - margin.left/2)
        .attr("x", (-height+margin.top)/2)
        .style("text-anchor", "end")
        .text("Mean Accuracy");

    // Add title
    canvas.append("text")
        .attr("x", (width / 2))
        .attr("y", 0 - (margin.top / 2))
        .attr("text-anchor", "middle")
        .style("font-size", "20px")
        .style("text-decoration", "underline")
        .text("Frequency and accuracy of selected features");

    // Add subtitle
    canvas.append("text")
        .attr("x", (width / 2))
        .attr("y", 0 - (margin.top / 2) + 20)
        .attr("text-anchor", "middle")
        .style("font-size", "10px")
        .text("A plot that shows how often a feature was selected and the average accuracy of the models it was used for. \n Hover over the dots to see the feature number and frequency. ");


    // Add the tooltip container to the vis container
    // It's invisible and its position/contents are defined during mouseover
    var tooltip = d3.select("#vis-container").append("div")
        .attr("class", "tooltip")
        .style("opacity", 0);

    // Tooltip mouseover event handler
    var tipMouseover = function(d) {
        var html  = "Frequency: " +d.freqs + "</br>" + "Feature: "+ d.features;

        tooltip.html(html)
            .style("left", (d3.event.pageX + 15) + "px")
            .style("top", (d3.event.pageY - 28) + "px")
          .transition()
            .duration(200)
            .style("opacity", .9)

    };

    // Tooltip mouseout event handler
    var tipMouseout = function(d) {
        tooltip.transition()
                .duration(300) // ms
                .style("opacity", 0);
    };

    // Add data points
    canvas.selectAll(".dot")
          .data(data)
          .enter()
          .append("circle")
          .attr("class", "dot")
          .attr("r", 3)
          // .transition()
          // .delay(function(d,i){return(i*3)})
          // .duration(2000)
          .attr("cx", function(d) { return xScale( d.freqs ); })     // x position
          .attr("cy", function(d) { return yScale( d.accs ); })  // y position
          .style("fill", function(d) { return colorScale(d.freqs); })

  canvas.selectAll(".dot")
        .on("mouseover", tipMouseover)
        .on("mouseout", tipMouseout)


};
