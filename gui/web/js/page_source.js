$(document).ready(function() {

    $("#time_header").on("click", function() {
        $("#time_panel").slideToggle(300);
    })

    $("#space_header").on("click", function() {
        $("#space_panel").slideToggle(300);
    })

    $("#plot_header").on("click", function() {
        $("#plot_panel").slideToggle(300);
    })

    $("#check_LBL").on("click", function() {eel.setFlags("reflection", false); drawMenu();})
    $("#check_TMM").on("click", function() {eel.setFlags("reflection",  true); drawMenu();})

    drawMenu()
    drawMaterial();

});

async function drawMaterial() {
    nindex = await eel.getIndexN()();
    reflection = await eel.getFlags("reflection")();

    let labels = [];
    
    if (reflection) {
        nindex.reduce(function(dummy, layer) {labels.push(layer.nr + " + " + layer.ni + "i");}, 0);
    } else {
        nindex.reduce(function(dummy, layer) {labels.push(layer.l);}, 0);
    }
    
    await drawMaterial_core(labels);
    $(".canvas > div").on("click", selectLayer);
};

async function drawMenu() {

    reflection = await eel.getFlags("reflection")();

    let content;
    console.log(reflection);

    if (reflection) {
        content =   "<tr>" +
                        "<td>Refractive Index (real)</td>" + 
                        "<td>Refractive Index (imag)</td>" +
                    "</tr>" +
                    "<tr><td><input></td><td><input></td></tr>";

    } else {
        content =   "<tr><td>Absorption Length</td></tr>" + 
                    "<tr><td><input></td></tr>";
    }

    $("#table_space tr").remove();
    $("#table_space").prepend(content);

} 

function selectLayer() {

}