$ ->
  $('.test-images a').click (e) ->
    e.stopPropagation()
    image = $(this).data('image-link')
    $(this).closest('.test-images').addClass('image-chosen')
    $(this).css('top', $(this).position().top).css('left', $(this).position().left)
    $(this).addClass('chosen')
    $(this).animate { top:0, left:'50%', marginLeft:'-150px' }, 1000, ->
      $(this).closest('.test-images').addClass('loading').find('a:not(.chosen)').hide()
      $('#start-over-link').removeClass('hidden')
      $.ajax({
        url: "/solve",
        method: "post",
        data: {
          image: image
        },
        success: (response) ->
          console.log(response)
          if response.indexOf("SOLVED") >= 0
            $('#results').load '/results?image='+image, ->
              $(this).closest('.test-images').removeClass('loading').addClass('loaded')
          else
            $("#errors").html('<p>Woops, we were unable to find a solution for that puzzle. Please try a different image.</p>')
            $('.test-images').removeClass('loading')
        ,
        error: (response) ->
          $("#errors").html('<p>Woops, something went wrong. Please try again.</p>')
          $('.test-images').removeClass('loading')
      })
