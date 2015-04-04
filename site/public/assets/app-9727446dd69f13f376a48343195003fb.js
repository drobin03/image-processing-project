(function() {
  $(function() {
    return $('.test-images a').click(function(e) {
      var image;
      e.stopPropagation();
      image = $(this).data('image-link');
      $(this).closest('.test-images').addClass('image-chosen');
      $(this).css('top', $(this).position().top).css('left', $(this).position().left);
      $(this).addClass('chosen');
      return $(this).animate({
        top: 0,
        left: '50%',
        marginLeft: '-150px'
      }, 1000, function() {
        $(this).closest('.test-images').addClass('loading');
        $('#start-over-link').removeClass('hidden');
        return $.ajax({
          url: "/solve",
          method: "post",
          data: {
            image: image
          },
          success: function(response) {
            console.log(response);
            if ($.trim(response) !== "") {
              return $('#results').load('/results?image=' + image, function() {
                return $(this).closest('.test-images').removeClass('loading').addClass('loaded');
              });
            } else {
              return $('.test-images').append('<p>Woops, we were unable to find a solution for that puzzle. Please try again.</p>').removeClass('loading');
            }
          }
        });
      });
    });
  });

}).call(this);
