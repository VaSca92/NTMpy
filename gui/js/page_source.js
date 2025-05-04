let layer_num = 0;
let source_set;
let reflection;
let nindex;
let source;

$(document).ready(function() {

    $("#time_header").on("click", function() {
        $("#time_panel").slideToggle(300);
        $("#plot_panel").slideUp(300);
    })

    $("#space_header").on("click", function() {
        $("#space_panel").slideToggle(300);
        $("#plot_panel").slideUp(300);
    })

    $("#plot_header").on("click", function() {
        $("#plot_panel").slideToggle(300);
        $( "#time_panel").slideUp(300);
        $("#space_panel").slideUp(300);
    })

    $("#check_LBL").on("click", function() {reflection = false; eel.setFlags("reflection", false); drawPage();})
    $("#check_TMM").on("click", function() {reflection =  true; eel.setFlags("reflection",  true); drawPage();})

    $("#update_space").on("click", modifyIndexN);
    $("#update_time").on("click", setSource);

    drawPage();

});

async function drawPage() {
    reflection = await eel.getFlags("reflection")();
    nindex = await eel.getIndexN()();
    source_set = await eel.getFlags("source_set")();

    let labels = [];
    
    if (reflection) {
        nindex.reduce(function(dummy, layer) {labels.push(layer.nr + " + " + layer.ni + "i");}, 0);
    } else {
        nindex.reduce(function(dummy, layer) {labels.push(layer.l);}, 0);
    }

    if (source_set||true) {
        source = await eel.getSource()();
        $("#table_time input:eq(0)").val(source.energy);
        $("#table_time input:eq(1)").val(source.fwhm);
        $("#table_time input:eq(2)").val(source.delay);
    }

    
    await drawMaterial_core(labels);
    $(".canvas > div").on("click", selectLayer);

    drawMenu();
    drawAxis();

};

function drawMenu() {

    let content;

    if (reflection) {
        content =   "<tr>" +
                        "<td>Refractive Index (real)</td>" + 
                        "<td>Refractive Index (imag)</td>" +
                    "</tr>" +
                    "<tr><td><input></td><td><input></td></tr>";
        $("#check_TMM").prop("checked", true);
    } else {
        content =   "<tr><td>Absorption Length</td></tr>" + 
                    "<tr><td><input></td></tr>";
        $("#check_LBL").prop("checked", true);
    }

    $("#table_space tr").remove();
    $("#table_space").prepend(content);

} 

function selectLayer() {
    layer_num = $(this).index();

    $("#space_panel div label:eq(0)").text("Properties of Layer " + layer_num);

    if (reflection) {
        $("#table_space input:eq(0)").val(nindex.nr);
        $("#table_space input:eq(1)").val(nindex.ni);
    } else {
        $("#table_space input:eq(0)").val(nindex.l);
    }

}


function setSource() {
    energy = $("#table_time input:eq(0)").val();
    fwhm   = $("#table_time input:eq(1)").val();
    delay  = $("#table_time input:eq(2)").val();

    complete = energy != "" && fwhm != "" && delay != ""; 

    eel.setSource(energy, fwhm, delay);
    eel.setFlags("source_set", true);

    $("#helpbar").css("color","#ffffff");
    $("#helpbar").text("Source added correctly");
}

function modifyIndexN() {
    console.log(layer_num)
    if (reflection) {
        nr = $("#table_space input:eq(0)").val();
        ni = $("#table_space input:eq(0)").val();
        eel.setIndexN( nr, ni, layer_num)
    } else {
        lambda = $("#table_space input:eq(0)").val();
        eel.setIndexN( lambda, 0, layer_num)
    }

    drawPage()
}