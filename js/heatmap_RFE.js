// To see page: https://annemijnd.github.io/codesample/html/heatmap_RFE.html

// Define some globals
var myFEATURES = [];
var myTYPES = [];

// Set the dimensions and margins of the graph
var margin = {top: 80, right: 80, bottom: 50, left: 100},
  width = 1200 - margin.left - margin.right,
  height = 450 - margin.top - margin.bottom;

// Define path of data and import data
var datapath = "../";
try {
  d3.csv(datapath + "results/features/heatmap_RFE_freq.csv", function(data) {
    console.log(data)
  })
}
catch(err) {
  datapath = "../../"
  d3.csv(datapath + "results/features/heatmap_RFE_freq.csv", function(data) {
    console.log(data)})
}


d3.csv(datapath+ "results/features/heatmap_RFE_freq.csv", function(data) {
  var feature_array = [];
  var type_array = [];
  var freq_array = [];

  // Loop through data
  data.forEach(function(entry) {
      if (!feature_array.includes(entry.features)){
          feature_array.push(entry.features)
      };

      if (!type_array.includes(entry.type)){
      type_array.push(entry.type)
      };

      if (!freq_array.includes(entry.freqs)){
        freq_array.push(entry.freqs)
      }
  });

  // Create svg
  var svg = d3.select("#my_dataviz")
              .append("svg")
              .attr("class", "datasvg")
              .attr("width", width + margin.left + margin.right)
              .attr("height", height + margin.top + margin.bottom)
              .append("g")
              .attr("transform",
                    "translate(" + margin.left + "," + margin.top + ")");

  // Labels of row and columns
  var myGroups = feature_array
  var myVars = type_array

  // Build X scales and axis:
  var x = d3.scaleBand()
            .range([ 0, width ])
            .domain(myGroups)
            .padding(0.01);

  svg.append("text")
      .attr("class", "label")
      .attr("y", height + margin.bottom/2)
      .attr("x", width/2 + 30)
      .style("text-anchor", "end")
      .text("Frequency (low to high)");

  // Build Y scales and axis:
  var y = d3.scaleBand()
            .range([ height, 0 ])
            .domain(myVars)
            .padding(0.01);

  svg.append("g")
      .call(d3.axisLeft(y))

  svg.append("text")
      .attr("class", "label")
      .attr("transform", "rotate(-90)")
      .attr("y",  -margin.left/2)
      .attr("x", (-height+margin.top+margin.bottom)/2)
      .attr("font", "30px")
      .style("text-anchor", "end")
      .text("Breast cancer subtype");


    // Add title
    svg.append("text")
        .attr("x", (width / 2))
        .attr("y", 0 - (margin.top / 2) - 10)
        .attr("text-anchor", "middle")
        .style("font-size", "22px")
        .style("text-decoration", "underline")
        .text("Frequency and accuracy of selected features per cancer subtype");


    // Add subtitle
    svg.append("text")
        .attr("x", (width / 2))
        .attr("y", -20)
        .attr("text-anchor", "middle")
        .style("font-size", "15px")
        .text("A plot that shows per subtype how often a feature was selected and what its accuracy was. " +
             "Each element is one feature.")

    svg.append("text")
        .attr("x", (width / 2))
        .attr("y", -5)
        .attr("text-anchor", "middle")
        .style("font-size", "15px")
        .text("The features are ordered by frequency. Hover over the elements to see information.")

  // Build color scale
  var myColor = d3.scaleLinear()
                  .range(["red", "green"])
                  .domain([0,1]);

  // Create tooltip
  var tooltip = d3.select("#my_dataviz")
                  .append("div")
                  .attr("class", "tooltip")
                  .attr("maxwidth", "10px")
                  .style("background-color", "white")
                  .style("border", "solid")
                  .style("border-width", "2px")
                  .style("border-radius", "5px")
                  .style("padding", "5px")


  // Create three functions that change the tooltip when user hover / move / leave a cell
  var mouseover = function(d) {
    tooltip
    d3.select(this)
      .style("stroke", "black")
  }

  var mousemove = function(d) {
    tooltip
      .html("Feature: " + d.features + " </b>" + " Frequency: " + d.freqs + "</b>" + " Accuracy: " + d.accuracy )
      .style("left", (d3.mouse(this[0] + 70) + "px"))
      .style("top", (d3.mouse(this)[1]) + "px")
  }

  var mouseleave = function(d) {
    tooltip
    d3.select(this)
      .style("stroke", "none")
  }

  // Create legend bar
  var colorScale =   d3.scaleLinear()
                    .range(["green", "red"])
                    .domain([0,1]);

  // Legend bar dimensions
  var legendWidth = 20;
  var legendHeight = 200;
  var legendX = width - margin.right + margin.right/2;
  var legendY = 300;
  var legendX = width + 30;
  var legendY = height - margin.top - margin.bottom - legendHeight /2;

  // Legend bar rectangle
  svg.append("rect")
      .attr("class", "legendbar")
      .attr("x", legendX)
      .attr("y", legendY)
      .attr("width", legendWidth)
      .attr("height", legendHeight)
      .style("fill", "url(#legendGradient)");

  // Define linear range and fill rectangle with gradient
  var defs = svg.append("defs");
  var linearGradient = defs.append("linearGradient")
      .attr("id", "legendGradient")
      .attr("x1", "0%")
      .attr("y1", "0%")
      .attr("x2", "0%")
      .attr("y2", "100%");

  linearGradient.append("stop")
      .attr("offset", "0%")
      .attr("stop-color", colorScale(0));

  linearGradient.append("stop")
      .attr("offset", "100%")
      .attr("stop-color", colorScale(1));

  // Add bar text
  svg.append("text")
      .attr("x", legendX + 25)
      .attr("y", legendY + 11)
      .text("1.0")

  svg.append("text")
      .attr("x", legendX + 25)
      .attr("y", legendY + legendHeight)
      .text("0.0");

  // Add legend title
  svg.append("text")
      .attr("transform", "rotate(90)")
      .attr("x", legendY/2 + legendHeight)
      .attr("y", -legendX - 30)
      .attr('font', "5px")
      .style("text-anchor", "end")
      .text("Mean accuracy");

  // Add data
  svg.selectAll()
      .data(data)
      .enter()
      .append("rect")
      .attr("class", "datapoint")
      .attr("x", function(d) { return x(d.features) })
      .attr("y", function(d) { return y(d.type) })
      .attr("width", x.bandwidth() )
      .attr("height", y.bandwidth() )
      .style("fill", function(d) { return myColor(+d.accuracy)} )
          .on("mouseover", mouseover)
          .on("mousemove", mousemove)
          .on("mouseleave", mouseleave);



})
